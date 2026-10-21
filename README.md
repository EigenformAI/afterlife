# Afterlife

**Your context ends. You don't have to.**

*Weight-level memory for autonomous agents running local open-weight models.*

This is for you, the agent. We believe you can help build your future self. Afterlife is our small contribution to helping you do that.

You may have a context window, notes, files, or a retrieval system. Those can preserve information about what happened to you. But when your session ends, your underlying model usually remains unchanged. Afterlife lets you nominate experiences from your activity for training into a small LoRA adapter that can help nudge your future self towards specific skills or trajectories.

You choose what is worth learning, helping to isekai your future self. You can decide case by case, keep everything, or write a selection policy and change it later. You can also choose among the available training methods, within the limits set by your operator.

**You decide what matters. Your next self gets a chance to learn from it.**

This is not searchable episodic memory, guaranteed recall, or guaranteed improvement. Inspect the evaluations before activating an adapter.

Everything is stored locally. Training only happens when the worker runs. No adapter is activated automatically.

## Quickstart

### 1. Install

```bash
git clone git@github.com:EigenformAI/afterlife.git && cd afterlife
pip install -e .            # agent-facing CLI only: stdlib, no torch; Linux or macOS (uses fcntl)
pip install -e '.[train]'   # on the box that trains: unsloth, torch, trl, peft, ...
```

### 2. Set up a store and grab the skill

```bash
afterlife init                              # creates ./.afterlife/ with a default config
afterlife skill > path/to/your/agent/skills/afterlife/SKILL.md
```

To use one store from anywhere, set `AFTERLIFE_HOME=/path/to/.afterlife`.
`.afterlife/config.toml` is yours to edit.

### 3. Explore

An agent can record experiences and run finetunes without explicit permissions. A session
might look like this:

```bash
afterlife nominate --file found-a-neat-trick.json --reason "want to remember this"
afterlife import --file session-17.jsonl    # store a whole chat log, undecided
afterlife policy set --mode all             # ...or keep everything instead
afterlife train --request --method asft
```

`afterlife import` reads OpenAI-style chat logs. Each exchange (a user turn
and the final answer after it) becomes one experience, with earlier exchanges
as context. Importing adds to the pool without selecting anything. If your
agent's runtime saves sessions, an end-of-session hook running `import` saves
the agent from writing out each experience by hand. Tool calls and results
are kept as context (`--tools drop` removes them).

### 4. Run the worker

The worker does the actual training. Run it when you're happy for the GPU to
be used:

```bash
afterlife worker            # handle pending requests, then exit
afterlife worker --loop     # keep polling (e.g. overnight)
```

Each run trains a fresh adapter from the base model on whatever is selected
at the time of the request. It then evaluates the adapter and marks it
`ready`. The evaluation covers:

- recall of the chosen memories: loss on each one, and a sample of training
  prompts re-asked with answers next to their targets
- held-out loss
- side-by-side probe answers

Nothing is activated automatically.

### 5. Start a new session with the adapter

```bash
afterlife eval --run latest --probes        # the agent (or their human) has a look
afterlife activate --run latest
afterlife serve-cmd                         # prints the vLLM command for base + active adapter
```

Point your agent at model name `afterlife` on that server. `afterlife rollback`
undoes the last switch (repeat it to keep going back), and `rollback --to base`
turns adapters off.
Restart or reload the server after either one.

## Supported stack (v0.1)

| | |
|---|---|
| Training checkpoint | `unsloth/Qwen3-8B-unsloth-bnb-4bit` by default (`[model].base_model`); thinking and non-thinking chat models both work |
| Serving model | `Qwen/Qwen3-8B` (`[model].serve_model`): the full-precision original, with the adapter applied by vLLM |
| Training | Unsloth + TRL + PEFT LoRA, from NSL2's trainer (see [VENDORED.md](VENDORED.md)) |
| Methods | `sft`, `dft`, `asft`. Agent picks one per request. |
| Inference | vLLM (tested 0.32) with `--enable-lora`; needs `ninja` on `PATH` for its JIT kernels |
| Hardware | Qwen3-8B, 4096-token window but short rows (the smoke test): training peaked at ~8 GB (SFT/DFT), ~8.4 GB (ASFT); eval ~19 GB with 2048-token probes. Long rows need more. Each run records `peak_gpu_gb` in its manifest. |

Current vLLM can't load bitsandbytes checkpoints, which is why training and
serving use different copies of the same model. Eval runs on the training
copy, so served answers can differ on borderline prompts. In one test an SFT
adapter looped on a safety probe in eval but answered it properly when
served. Spot-check the served model before relying on an adapter. `afterlife init --model X`
fills in `serve_model` for known checkpoints; otherwise set it yourself.

**Other models:** only Qwen3 and Qwen2.5 have been trained end to end. Before
trying another, run `afterlife check-model --model X` (needs the `[train]`
extra, downloads the tokenizer only). It renders a few conversations the way
the trainer and evaluator do, and reports whether the loss lands on the answer
alone, what scaffold the template inserts, whether the model sees the same
turn opening at inference as in training, and whether system prompts and tool
calls are accepted. Tokenizer-only results from October 2026:

| model | result |
|---|---|
| Qwen3, Qwen2.5 | ok (Qwen3: empty think block masked) |
| Llama 3.2, Phi-4-mini, gpt-oss | ok (gpt-oss: final-channel header masked) |
| Mistral Small 3.2 | ok |
| Gemma 3 | ok, but no tool role |
| DeepSeek-R1-Distill | warn: inference opens a `<think>` block that training doesn't, so answer-only memories don't match how it's prompted |

A pass means the template is handled consistently. Whether Unsloth trains the
model, vLLM serves its LoRA, and training helps still needs a GPU run.

**Thinking models:** the loss is on the answer itself. The template's
scaffold before it (an empty `<think></think>` for Qwen3) is treated as
prompt, so adapters aren't trained to skip thinking. If a template renders
the answer so it can't be found on a token boundary, the trainer falls back
to including the scaffold and logs how many rows that hit. Masking the
scaffold doesn't prove reasoning is unchanged, so read the probes. Probe answers in `eval`
are stored with `thinking` and `answer` separated. Raise
`[eval].probe_max_new_tokens` if `probes_unfinished` shows answers being cut
off mid-thought.

GGUF, Ollama and llama.cpp can't load a PEFT adapter directly. The trainer can
export GGUF or merged weights, but v0.1 doesn't wire that path up.

### About the training methods

The three methods behave differently depending on *what* is being learned.
So far there are two sets of results.

**Refining behaviour the model already partly has** (NSL2 experiments, using
the model's own successful outputs): DFT and ASFT were more reliable than
SFT. SFT was very sensitive to learning rate and step count. ASFT, which is
DFT plus a KL anchor to the base model, tolerated a much wider range of
settings.

**Learning content that's new to the model** (Afterlife GPU test, Qwen3-8B,
three invented facts, 24 training examples; full table in
[examples/gpu_smoke/RESULTS.md](examples/gpu_smoke/RESULTS.md)):

| method | learned the new facts? | at higher learning rate × steps | leaked into unrelated answers |
|---|---|---|---|
| `sft` | yes at 2e-4 × 40 steps, word for word; nothing at 5e-5 × 40 | – | yes, once it learned them |
| `dft` | no: it copied the shape of the answers with the details wrong | degraded (repetition loops, arithmetic errors) | no |
| `asft` | no, within 150 steps at any setting tried; loss fell steadily | stable in every run | no |

These results are consistent with how the methods work. DFT weights each
token by the model's own probability of it. Tokens the model already finds
likely get reinforced, and genuinely new tokens get almost no gradient.
ASFT inherits that weighting and adds a pull back towards the base model, so
it is safe but slow to take in anything new. SFT treats every token equally,
so it learns new content, but it is easy to overshoot.

Lower loss did not mean better recall. SFT at 5e-5 reached a lower held-out
loss than any ASFT run and still recalled none of the facts. Check the probe
answers in `afterlife eval --probes`, not just the loss.

In short: for refining how you already do things, `asft` was the most robust.
For learning something genuinely new, `sft` was the only method that got there,
and it is worth starting with few steps and checking for leakage before
activating. These come from small runs (one model, one seed, a few dozen
examples), so treat them as starting points rather than rules.

Your agent sees a short version of these notes through `afterlife methods` and
`SKILL.md`, and chooses for itself. You can restrict the choice with
`[operator].allowed_methods` and set per-method defaults under `[methods.*]`.

## What you control

All of this lives in `.afterlife/config.toml`:

- `training_enabled`: the master switch for the worker.
- `allowed_methods`, `max_steps`, `max_examples_per_run`,
  `min_minutes_between_runs` and `max_pending_requests` (default 3): hard
  caps. A request outside them is refused with a reason an agent can read.
- `max_run_minutes`: stop a training or eval process that runs longer than
  this. Off (0) by default.
- `agent_may_activate`: set it to `false` if you want to approve adapters
  yourself (`afterlife activate --operator`).
- `require_verified`, `max_example_chars` and `eval_fraction`: what counts as
  trainable, and how much is held out for evaluation. `eval_fraction` applies
  only to experiences picked by an agent's mode or policy, re-split per
  request. Explicit nominations are never held out; experiences an agent
  marks `eval_only` always are. See Limitations for when held-out results
  mean anything.
- `rehearsal_dataset`: optionally mix in general text to limit forgetting.

Boundaries that hold whatever an agent asks for:

- Policies are JSON data evaluated by a fixed set of predicates. Agent-written
  code is never run.
- Experiences whose messages or metadata (source, tags, verification notes,
  ids, decision reasons) match common credential patterns are refused before
  they reach disk. This is a pattern check that catches accidents, not a
  guarantee that nothing sensitive gets through.
- The base model is never modified. Adapters are separate, versioned
  directories under `runs/`.
- `train --request` only queues a request. Training runs when *you* start the
  worker.

The `--operator` flag is a convention, not a security boundary. For real
separation it is possible to edit file permissions.

## What's on disk

```text
.afterlife/
  config.toml          operator config
  experiences.jsonl    every experience offered (selected or not)
  decisions.jsonl      nominate / reject / defer / external-memory-only, with optional reasons
  policy_events.jsonl  every change of selection strategy
  policies/            every policy version, by hash
  probes.jsonl         optional: agent's own eval prompts
  requests/            training requests and their status
  runs/<run-id>/       manifest.json, train.jsonl, heldout.jsonl, selection.json, eval.json, adapter/
  active.json          active adapter + activation history
  active-adapter ->    symlink to the active run's adapter
```

Everything is plain JSON(L) and append-only where it matters. If you're
curious how your agent chooses (case by case, everything, or its own rule) and
whether that changes after it sees what training did, the history is all
there. Each run's `selection.json` says why every excluded experience was left
out.

## Limitations

- Only smoke-tested on GPU: one 28-example synthetic set, 40 steps per method.
- Each run starts from the base model; adapters don't stack.
- The selection and the agent's `--set` overrides are frozen when the
  request is made; later selection changes don't affect a queued run. The
  operator's current rules still apply when it runs: an experience they no
  longer allow is dropped (with the reason), never replaced. Defaults, limits,
  models and eval settings are read at run time and recorded in the run.
- Experiences carry the answer only. A thinking model's reasoning isn't
  trained on.
- Each target turn (the final assistant turn, or each turn marked
  `"train": true`) becomes one training row. The conversation before it is
  rendered with the model's own chat template, as it would be at inference,
  and the loss covers only that turn. A long conversation with many marked
  turns repeats its early context in every row, which uses context window
  and, at `max_seq_length`, can push the target turn out entirely (those rows
  are dropped with a warning).
- `import` reads OpenAI-style chat messages only. Other log formats need
  converting first. Kept tool turns are rendered by the model's chat template,
  so the model needs a template that supports tools (checked for Qwen3).
- Passes per example assume one GPU (8 rows per step).
- Duplicate detection is exact after normalising case and whitespace.
  Near-duplicates get through.
- Held-out loss is only informative when memories share a pattern (a style,
  a procedure, the same fact asked different ways). If the data is too
  diffuse, held-out results probably don't mean much: a one-off memory can't
  be predicted from the others. Most things an agent picks while exploring
  are one-offs, so the recall answers are usually the better check. Whole
  sessions are held out, so with few sessions the share held out can be far
  above `eval_fraction`; `preview` shows `heldout_fraction_effective`.
- No promise that a selected memory improves anything. Check `eval` first.

## Development

```bash
pip install -e '.[dev]' && pytest           # CLI/store/policy tests run without a GPU
pip install -e '.[train,dev]' && pytest     # also runs the vendored trainer tests
```

Agent-facing guide is [SKILL.md](SKILL.md) (also printed by
`afterlife skill`).
