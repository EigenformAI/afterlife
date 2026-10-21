"""check-model against fake chat templates (one character per token)."""

from afterlife.modelcheck import check_tokenizer


class TemplateTokenizer:
    """ChatML-like template. ``scaffold`` goes before an assistant answer in
    training renders; ``gen_suffix`` ends the generation prompt."""

    chat_template = "fake"
    bos_token = None

    def __init__(self, scaffold="", gen_suffix="", answer_case=str):
        self.scaffold, self.gen_suffix, self.answer_case = scaffold, gen_suffix, answer_case

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **_kwargs):
        text = ""
        for m in messages:
            content = m["content"]
            if m["role"] == "assistant":
                content = self.scaffold + self.answer_case(content)
            text += f"<{m['role']}>{content}</{m['role']}>"
        return text + (f"<assistant>{self.gen_suffix}" if add_generation_prompt else "")

    def __call__(self, text=None, **_kwargs):
        return {"input_ids": [ord(c) for c in text], "attention_mask": [1] * len(text)}

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def test_a_consistent_template_passes_and_shows_its_scaffold():
    report = check_tokenizer(TemplateTokenizer(scaffold="<think></think>"), "some/model")
    assert report["status"] == "ok"
    case = report["cases"]["single_turn"]
    assert case["trained_text"] == "The answer is 4.</assistant>"
    assert case["scaffold"] == "<think></think>"


def test_a_preopened_think_block_is_flagged():
    report = check_tokenizer(TemplateTokenizer(gen_suffix="<think>"), "some/model")
    assert report["status"] == "warn"
    assert "not a prefix" in report["cases"]["single_turn"]["problem"]


def test_a_template_that_rewrites_the_answer_fails():
    report = check_tokenizer(TemplateTokenizer(answer_case=str.upper), "some/model")
    assert report["status"] == "fail"
    assert "fall back" in report["problems"][0]


def test_a_quantized_model_without_a_known_original_is_flagged():
    report = check_tokenizer(TemplateTokenizer(), "someone/model-bnb-4bit")
    assert any("serve_model" in w for w in report["warnings"])
