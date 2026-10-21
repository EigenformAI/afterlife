"""Operator-side check of a model's chat template, before spending GPU time on it.

Loads only the tokenizer (no weights, no GPU) and renders a few conversations
the way the trainer and evaluator will:

* mask: can the loss be put on the answer alone, or does the trainer fall back
  to training on the template's assistant-turn scaffold too?
* scaffold: what the template inserts between the generation prompt the model
  sees at inference and the answer (Qwen3: an empty think block). Masked, but
  worth knowing: the model is never shown it at inference.
* generation prompt: is what the model sees at inference a prefix of what it
  was trained on?
* special tokens: does the template render BOS itself (the trainer and
  evaluator encode rendered text without adding special tokens)?
* system prompts and tool calls: does the template accept them?

Passing says the template is handled consistently. It says nothing about
whether Unsloth can train the model, vLLM can serve its LoRA, or training
helps; that still needs a GPU run (examples/gpu_smoke).
"""

from __future__ import annotations

from typing import Any

from afterlife.config import KNOWN_SERVE_MODELS

ANSWER = "The answer is 4."
CASES: dict[str, list[dict[str, Any]]] = {
    "single_turn": [{"role": "user", "content": "What is 2 + 2?"}],
    "system_and_history": [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Hi."},
        {"role": "assistant", "content": "Hello."},
        {"role": "user", "content": "What is 2 + 2?"},
    ],
    "tool_call": [
        {"role": "user", "content": "Add 2 and 2 with the calculator."},
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "id": "call12345", "function": {"name": "add", "arguments": {"a": 2, "b": 2}}}]},
        {"role": "tool", "tool_call_id": "call12345", "name": "add", "content": "4"},
    ],
}
OPTIONAL_CASES = ("system_and_history", "tool_call")  # a template may legitimately lack these
TOOLS = {"tool_call": [{"type": "function", "function": {
    "name": "add", "description": "Add two numbers.",
    "parameters": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                   "required": ["a", "b"]}}}]}


def _render(tokenizer: Any, messages: list[dict[str, Any]], **kwargs: Any) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)


def check_case(tokenizer: Any, context: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    from afterlife.trainer.qlora import _answer_only_sft_row, _template_kwargs

    extra = _template_kwargs(tools)
    try:
        full = _render(tokenizer, context + [{"role": "assistant", "content": ANSWER}],
                       add_generation_prompt=False, **extra)
        prompt = _render(tokenizer, context, add_generation_prompt=True, **extra)
    except Exception as exc:  # noqa: BLE001 - templates raise anything from jinja errors to ValueError
        return {"status": "fail", "problem": f"template refused it: {type(exc).__name__}: {exc}"}

    out: dict[str, Any] = {"status": "ok"}
    row = _answer_only_sft_row(tokenizer, raw_response=ANSWER, context=context, tools=tools)
    if tools:
        out["tools_rendered"] = "add" in prompt and "Add two numbers" in prompt
    if row is None:
        out.update(status="fail", problem="answer not found on a token boundary: the trainer would fall back "
                                          "and also train on the assistant-turn scaffold")
    else:
        text_tok = getattr(tokenizer, "tokenizer", tokenizer)
        trained = text_tok.decode([t for t, m in zip(row["input_ids"], row["completion_mask"]) if m])
        out["trained_text"] = trained
        if not trained.startswith(ANSWER):
            out.update(status="fail", problem="loss covers more than the answer and its end-of-turn marker")

    start = full.rfind(ANSWER)
    if full.startswith(prompt):
        out["scaffold"] = full[len(prompt):start]  # inserted by training renders only; masked
    else:
        out["scaffold"] = None
        if out["status"] == "ok":
            out["status"] = "warn"
        out["problem"] = ("the generation prompt is not a prefix of the training render, so the model "
                          "sees a different turn opening at inference than in training")
        out["generation_prompt_tail"] = prompt[-80:]
        out["training_render_tail"] = full[max(0, start - 80):start]
    return out


def check_tokenizer(tokenizer: Any, base_model: str = "", serve_model: str | None = None) -> dict[str, Any]:
    if not getattr(tokenizer, "chat_template", None) and not getattr(
            getattr(tokenizer, "tokenizer", None), "chat_template", None):
        return {"status": "fail", "problems": ["no chat template"], "cases": {}}

    cases = {name: check_case(tokenizer, context, TOOLS.get(name)) for name, context in CASES.items()}
    problems, warnings = [], []
    for name, result in cases.items():
        if result["status"] == "fail" and name not in OPTIONAL_CASES:
            problems.append(f"{name}: {result['problem']}")
        elif result["status"] != "ok":
            warnings.append(f"{name}: {result['problem']}")

    text_tok = getattr(tokenizer, "tokenizer", tokenizer)
    bos = getattr(text_tok, "bos_token", None)
    rendered = _render(tokenizer, CASES["single_turn"], add_generation_prompt=True)
    special = {
        "bos_token": bos,
        "template_renders_bos": bool(bos) and rendered.startswith(bos),
        "tokenizer_adds_bos_by_default": bool(bos) and text_tok("x")["input_ids"][:1] == [text_tok.bos_token_id],
        "note": "Rendered text is encoded without adding special tokens, in training and in eval.",
    }
    if bos and not special["template_renders_bos"] and special["tokenizer_adds_bos_by_default"]:
        warnings.append("the tokenizer expects a BOS token the template doesn't render; "
                        "training and eval will run without it")

    serve = serve_model or KNOWN_SERVE_MODELS.get(base_model)
    if serve is None and any(q in base_model.lower() for q in ("bnb", "4bit", "8bit")):
        warnings.append("quantized checkpoint with no known full-precision original: set [model].serve_model, "
                        "since vLLM can't load bitsandbytes checkpoints")
    status = "fail" if problems else "warn" if warnings else "ok"
    return {"status": status, "problems": problems, "warnings": warnings, "serve_model": serve,
            "special_tokens": special, "cases": cases,
            "not_checked": "Unsloth loading, vLLM LoRA serving, and whether training helps: needs a GPU run."}


def check_model(base_model: str, serve_model: str | None = None) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    return {"model": base_model, **check_tokenizer(tokenizer, base_model, serve_model)}
