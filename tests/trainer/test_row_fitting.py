"""Row building under a token budget: long rows lose old turns, not their answer.

Torch-free (one character per token), so these run wherever loguru is installed.
"""

from afterlife.trainer.qlora import _drop_oldest_turn, _fit_sft_row


class ChatTokenizer:
    def __init__(self, rewrite_answer=False):
        self.rewrite_answer = rewrite_answer

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **_kwargs):
        text = ""
        for i, m in enumerate(messages):
            content = m["content"]
            if self.rewrite_answer and i == len(messages) - 1 and m["role"] == "assistant":
                content = content.upper()  # answer no longer findable: forces the NSL fallback
            text += f"<{m['role']}>{content}</{m['role']}>"
        return text

    def __call__(self, text=None, **_kwargs):
        return {"input_ids": [ord(c) for c in text], "attention_mask": [1] * len(text)}


def _convo(n_exchanges, system=True):
    msgs = [{"role": "system", "content": "be brief"}] if system else []
    for i in range(n_exchanges):
        msgs += [{"role": "user", "content": f"question {i} " + "x" * 40},
                 {"role": "assistant", "content": f"answer {i}"}]
    return msgs + [{"role": "user", "content": "last question"}]


def _trained(row):
    return "".join(chr(t) for t, m in zip(row["input_ids"], row["completion_mask"]) if m)


def test_a_row_that_fits_is_unchanged():
    stats = {}
    row = _fit_sft_row(ChatTokenizer(), prompt=None, raw_response="final", context=_convo(1),
                       max_seq_length=1000, stats=stats)
    assert _trained(row) == "final</assistant>" and stats == {}


def test_a_long_row_drops_old_turns_and_keeps_the_whole_answer():
    stats = {}
    context = _convo(6)
    row = _fit_sft_row(ChatTokenizer(), prompt=None, raw_response="final", context=context,
                       max_seq_length=200, stats=stats)
    text = "".join(chr(t) for t in row["input_ids"])
    assert len(row["input_ids"]) <= 200
    assert text.startswith("<system>be brief</system><user>")  # system kept, starts on a user turn
    assert "last question" in text and _trained(row) == "final</assistant>"
    assert "question 0" not in text and stats == {"context_trimmed": 1}
    assert len(context) == 14  # the caller's list is not modified


def test_a_row_that_cannot_fit_is_dropped_and_counted():
    stats = {}
    row = _fit_sft_row(ChatTokenizer(), prompt=None, raw_response="final " * 50, context=_convo(2),
                       max_seq_length=100, stats=stats)
    assert row is None and stats == {"too_long": 1}


def test_an_unmaskable_row_is_skipped_not_fatal():
    class Unprefixable(ChatTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            text = super().apply_chat_template(messages, **kwargs)
            return ("[last]" if messages[-1]["role"] == "user" else "[asst]") + text  # prompt isn't a prefix

    stats = {}
    row = _fit_sft_row(Unprefixable(rewrite_answer=True), prompt=None, raw_response="final",
                       context=_convo(1), max_seq_length=1000, stats=stats)
    assert row is None and stats == {"unmaskable": 1}


def test_mask_fallback_is_counted():
    stats = {}
    row = _fit_sft_row(ChatTokenizer(rewrite_answer=True), prompt=None, raw_response="final",
                       context=_convo(1), max_seq_length=1000, stats=stats)
    assert row is not None and stats == {"mask_fallback": 1}


def test_drop_oldest_turn_never_removes_the_last_user_turn():
    context = [{"role": "user", "content": "only"}]
    assert not _drop_oldest_turn(context) and context == [{"role": "user", "content": "only"}]
    context = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b", "tool_calls": []},
               {"role": "tool", "content": "c"}, {"role": "user", "content": "d"}]
    assert _drop_oldest_turn(context) and context == [{"role": "user", "content": "d"}]


def test_tool_definitions_reach_the_chat_template():
    seen = []

    class Recording(ChatTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            seen.append(kwargs.get("tools"))
            return super().apply_chat_template(messages, **kwargs)

    tools = [{"type": "function", "function": {"name": "add"}}]
    _fit_sft_row(Recording(), prompt=None, raw_response="final", context=_convo(1),
                 max_seq_length=1000, stats={}, tools=tools)
    assert seen and all(t == tools for t in seen)
