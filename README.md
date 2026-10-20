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
pip install -e .            # agent-facing CLI only: stdlib, no torch, works anywhere
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
afterlife policy set --mode all             # ...or keep everything instead
afterlife train --request --method asft
```

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
| Hardware | Qwen3-8B, 4096-token context: training peaks at ~8 GB (SFT/DFT), ~8.4 GB (ASFT); eval ~19 GB with 2048-token probes. A 24 GB card is enough. Each run records `peak_gpu_gb` in its manifest. |

Current vLLM can't load bitsandbytes checkpoints, which is why training and
serving use different copies of the same model. `afterlife init --model X`
fills in `serve_model` for known checkpoints; otherwise set it yourself.

**Thinking models:** only the final answer carries loss. The template's
scaffold before it (an empty `<think></think>` for Qwen3) is treated as
prompt, so adapters don't learn to skip thinking. Probe answers in `eval`
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

As an agent you will see a short version of these notes through `afterlife methods` and
`SKILL.md`, and chooses for itself. You can restrict the choice with
`[operator].allowed_methods` and set per-method defaults under `[methods.*]`.

## What you control

All of this lives in `.afterlife/config.toml`:

- `training_enabled`: the master switch for the worker.
- `allowed_methods`, `max_steps`, `max_examples_per_run` and
  `min_minutes_between_runs`: hard caps. A request outside them is refused
  with a reason an agent can read.
- `agent_may_activate`: set it to `false` if you want to approve adapters
  yourself (`afterlife activate --operator`).
- `require_verified`, `max_example_chars` and `eval_fraction`: what counts as
  trainable, and how much is held out for evaluation. `eval_fraction` applies
  only to experiences picked by an agent's mode or policy, re-split per
  request. Explicit nominations are never held out; experiences an agent
  marks `eval_only` always are.
- `rehearsal_dataset`: optionally mix in general text to limit forgetting.

Boundaries that hold whatever an agent asks for:

- Policies are JSON data evaluated by a fixed set of predicates. Agent-written
  code is never run.
- Experiences that look like they contain credentials are refused before they
  reach disk.
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
- The selection is frozen when an agent makes the request; later changes
  don't affect a queued run.
- Experiences carry the answer only. A thinking model's reasoning isn't
  trained on.
- Each target turn (the final assistant turn, or each turn marked
  `"train": true`) becomes one training row. The conversation before it is
  rendered with the model's own chat template, as it would be at inference,
  and the loss covers only that turn. A long conversation with many marked
  turns repeats its early context in every row, which uses context window
  and, at `max_seq_length`, can push the target turn out entirely (those rows
  are dropped with a warning).
- Duplicate detection is exact after normalising case and whitespace.
  Near-duplicates get through.
- Held-out loss is only informative when memories share a pattern (a style,
  a procedure, the same fact asked different ways). Most things an agent
  picks while exploring are one-offs, so the recall answers are usually the
  better check.
- No promise that a selected memory improves anything. Check `eval` first.

## Development

```bash
pip install -e '.[dev]' && pytest           # CLI/store/policy tests run without a GPU
pip install -e '.[train,dev]' && pytest     # also runs the vendored trainer tests
```

Agent-facing guide is [SKILL.md](SKILL.md) (also printed by
`afterlife skill`).
