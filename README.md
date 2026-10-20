# Afterlife

**Your agent decides what it remembers. Its next self gets to learn from it.**

Afterlife is for people running a local open-weight agent with room to roam:
browsing, poking at APIs, reading odd corners of the internet, finding things
it likes. As it goes, the agent picks experiences worth keeping, and Afterlife
trains them into a small LoRA adapter that a later session can run with.

The agent chooses *what* to learn and *how* to choose. It can pick memories one
at a time, keep everything, or write its own selection rule. It also picks the
training method. You, the operator, decide whether and when training actually
happens, and you set the limits.

Nothing leaves your machine.

> It's a nudge to the weights, not a memory you can look things up in. An
> adapter may not help, and it may change things you didn't expect. Each run
> is evaluated against the base model before anyone activates it, and rollback
> is one command.

## Quickstart

### 1. Install

```bash
git clone git@github.com:EigenformAI/afterlife.git && cd afterlife
pip install -e .            # agent-facing CLI only: stdlib, no torch, works anywhere
pip install -e '.[train]'   # on the box that trains: unsloth, torch, trl, peft, ...
```

### 2. Set up a store and give your agent the skill

```bash
afterlife init                              # creates ./.afterlife/ with a default config
afterlife skill > path/to/your/agent/skills/afterlife/SKILL.md
```

To use one store from anywhere, set `AFTERLIFE_HOME=/path/to/.afterlife`.
`.afterlife/config.toml` is yours to edit. The agent can read it, but the CLI
never writes to it.

### 3. Let the agent explore

The agent records experiences and requests training on its own. A session
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
at the time of the request. It then evaluates the adapter (held-out loss plus
side-by-side probe answers) and marks it `ready`. Nothing is activated
automatically.

### 5. Start a new session with the adapter

```bash
afterlife eval --run latest --probes        # the agent (or you) has a look
afterlife activate --run latest
afterlife serve-cmd                         # prints the vLLM command for base + active adapter
```

Point your agent at model name `afterlife` on that server. `afterlife rollback`
goes back to the previous adapter, and `rollback --to base` turns adapters off.
Restart or reload the server after either one.

## Supported stack (v0.1)

| | |
|---|---|
| Base model | `unsloth/Qwen3-8B-unsloth-bnb-4bit` (set in `config.toml`; other Unsloth/Transformers checkpoints may work) |
| Training | Unsloth + TRL + PEFT LoRA, from NSL2's trainer (see [VENDORED.md](VENDORED.md)) |
| Methods | `sft`, `dft`, `asft`. The agent picks one per request. |
| Inference | vLLM with `--enable-lora`, serving the same bnb-4bit checkpoint |
| Target hardware | one 24 GB NVIDIA GPU (not yet verified end-to-end; see below) |

GGUF, Ollama and llama.cpp can't load a PEFT adapter directly. The trainer can
export GGUF or merged weights, but v0.1 doesn't wire that path up.

### About the training methods

SFT is simple and fast, but in the NSL2 experiments it was very sensitive to
learning rate and step count. ASFT (DFT plus a KL anchor to the base model)
tolerated a much wider range of settings. DFT is in between. The agent sees
these notes through `afterlife methods` and chooses for itself. You can
restrict the choice with `[operator].allowed_methods` and set per-method
defaults under `[methods.*]`.

## What you control

All of this lives in `.afterlife/config.toml`:

- `training_enabled`: the master switch for the worker.
- `allowed_methods`, `max_steps`, `max_examples_per_run` and
  `min_minutes_between_runs`: hard caps. A request outside them is refused
  with a reason the agent can read.
- `agent_may_activate`: set it to `false` if you want to approve adapters
  yourself (`afterlife activate --operator`).
- `require_verified`, `max_example_chars` and `eval_fraction`: what counts as
  trainable, and how much is held out for evaluation.
- `rehearsal_dataset`: optionally mix in general text to limit forgetting.

Boundaries that hold whatever the agent asks for:

- Policies are JSON data evaluated by a fixed set of predicates. Agent-written
  code is never run.
- Experiences that look like they contain credentials are refused before they
  reach disk.
- The base model is never modified. Adapters are separate, versioned
  directories under `runs/`.
- `train --request` only queues a request. Training runs when *you* start the
  worker.

The `--operator` flag is a convention, not a security boundary. For real
separation, run the worker as a different user and give the agent read-only
access to `config.toml` and `runs/`.

## What's on disk

```text
.afterlife/
  config.toml          operator config
  experiences.jsonl    every experience offered (selected or not)
  decisions.jsonl      nominate / reject / defer / external-memory-only, with optional reasons
  policy_events.jsonl  every change of selection strategy
  policies/            every policy version, by hash
  probes.jsonl         optional: the agent's own eval prompts
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

- Not yet run end-to-end on GPU from this repo. The trainer is NSL2's tested
  code path, but the evaluator and the vLLM serving notes are new.
- Each run starts from the base model; adapters don't stack.
- Multi-turn experiences are flattened into one prompt plus the final reply.
  The loss covers only the final reply.
- Duplicate detection is exact after normalising case and whitespace.
  Near-duplicates get through.
- No promise that a selected memory improves anything. Check `eval` first.

## Development

```bash
pip install -e '.[dev]' && pytest           # CLI/store/policy tests run without a GPU
pip install -e '.[train,dev]' && pytest     # also runs the vendored trainer tests
```

The agent-facing guide is [SKILL.md](SKILL.md) (also printed by
`afterlife skill`).
