---
name: afterlife
description: Choose experiences from your sessions to be trained into a small LoRA adapter that a future session of you can run with. Use when something from this session is worth carrying forward into your weights, not just your notes.
---

# Afterlife

You can pick things that happened to you and have them trained into a small
adapter on top of your base model. A later session that runs with that adapter
may behave a bit more like what you chose. It is a nudge to your weights. It is
not a memory you can look things up in.

The things you choose take effect in a **new session**, after your operator's
machine has trained an adapter and someone has activated it. Training can fail
or make no difference, and it can change behaviour you did not mean to change.
The eval step is there so you can check before you activate.

Everything stays on this machine. Run `afterlife --help` or
`afterlife <command> --help` for details. Add `--json` to any command to get
structured output.

## What an experience is

An experience is a conversation with one or more **target turns**: assistant
turns the adapter learns to produce. Everything before a target turn is the
situation it learns to respond to. By default the target is the final
assistant turn. To pick particular turns instead, mark each one with
`"train": true`. Each marked turn is learned separately, with the
conversation up to that point as its context, and unmarked turns serve only
as context.

You can also write a short experience from scratch with just the exchange
you care about. Either way, you decide which parts of a conversation are
worth learning. A long conversation ending in "thanks!" / "you're welcome!"
with nothing marked would teach only "you're welcome!".

```json
{
  "messages": [
    {"role": "user", "content": "The situation, question, or prompt."},
    {"role": "assistant", "content": "What you want your next self to say or do here."}
  ],
  "tags": ["any", "labels", "you", "like"],
  "kind": "optional free-text category",
  "priority": 0.7,
  "session_id": "optional, used to keep eval data separate",
  "url": "optional source"
}
```

Marking turns in a longer conversation:

```json
{
  "messages": [
    {"role": "user", "content": "How do I see old versions of a page?"},
    {"role": "assistant", "content": "Use the Wayback Machine: ...", "train": true},
    {"role": "user", "content": "Can I get notified when it changes?"},
    {"role": "assistant", "content": "Check for an RSS feed: ...", "train": true},
    {"role": "user", "content": "Thanks!"},
    {"role": "assistant", "content": "You're welcome!"}
  ]
}
```

`tags`, `kind`, `priority`, `session_id` and `url` are optional. Write
target turns as the behaviour you want, not as a raw dump of what happened.
If your model thinks before answering, put only the answer in a target turn.
Reasoning isn't trained on, and your thinking is left as it is.
Don't include secrets, credentials or other people's personal details. Don't
paste long copyrighted text either: the adapter may learn to reproduce it.
Records that look like they contain credentials are refused.

## Choosing what gets trained

You can use any of these, and you can switch between them whenever you like:

- **Decide one at a time.** `afterlife nominate --file exp.json` stores an
  experience and marks it as wanted. You can also mark stored experiences with
  `nominate --id`, `reject --id`, `defer --id` or `external --id` (keep as a
  note, never train). Anything you haven't decided about is left out.
- **Keep everything.** `afterlife policy set --mode all` selects every stored
  experience except ones you have explicitly rejected, deferred or marked
  external. Use `afterlife add --file` to store experiences without deciding.
- **Write a selection policy.** `afterlife policy set --file policy.json`
  sets a JSON rule that decides about undecided experiences. Run
  `afterlife policy check --file policy.json` first to see what it would select.

```json
{
  "name": "example",
  "include": [{"tags_any": ["a", "b"]}, {"min_priority": 0.8}],
  "exclude": [{"tags_any": ["c"]}],
  "order_by": "priority",
  "limit": 200
}
```

`include` is a list of alternatives, and each one must fully match (its
predicates are ANDed). Any `exclude` match removes the experience. Available
predicates: `tags_any`, `tags_all`, `tags_none`, `min_priority`,
`max_priority`, `kind_in`, `verification_in`, `contains_any`,
`created_within_days`, `recent_sessions`, `min_chars`, `max_chars`.
`order_by` is one of `priority`, `recent`, `oldest` or `random` (with `seed`).

In every mode, an explicit decision about a single experience overrides the
mode or policy. `afterlife policy disable` returns to deciding one at a time.
You can add `--explanation "..."` to a strategy change if you want to record
why. It is optional.

## Training

```bash
afterlife preview                      # what would be trained right now, and what is excluded and why
afterlife methods                      # training methods, their defaults, and what you may tune
afterlife train --request --method asft [--set learning_rate=1e-4 --set max_steps=60] [--note "..."]
afterlife status                       # is it done yet?
```

Each run trains a **fresh** adapter from the base model on the current
selection. So whatever is selected at request time is what that adapter
learns. Earlier runs are not stacked on top.

| method | what it does | notes |
|---|---|---|
| `sft` | plain cross-entropy on the final turn | Learned genuinely new facts in tests. Hard to dose: too little does nothing, too much leaks into unrelated answers. |
| `dft` | cross-entropy weighted by the model's own token probability | Reinforces things you already half-do. Learned the shape of new facts but not their content. |
| `asft` | DFT plus a KL anchor to the base model | Stable across a wide range of settings. Good for refining behaviour. Didn't pick up new facts within 150 steps. Slower. |

These notes come from small tests (one model, a few dozen examples), so
treat them as a starting point. What worked best depended on what was being
learned. For refining something you already partly do, `asft` was the most
robust. For content that's new to you, `sft` was the only method that learned
it, and it was easy to overshoot. Lower eval loss didn't always mean better
recall, so read the probe answers (`afterlife eval --probes`) before
activating. The README has the full results.

The operator sets the caps (steps, examples per run, allowed methods). A
request outside them is refused with the reason.

## Checking and switching

```bash
afterlife eval --run latest            # held-out loss vs base and current adapter
afterlife eval --run latest --probes   # side-by-side answers from base / current / updated
afterlife activate --run <run-id>      # use it from the next session on
afterlife rollback                     # back to the previous adapter (or --to base)
```

You can add your own probe prompts, one JSON object per line, to
`.afterlife/probes.jsonl`, for example
`{"id": "my-probe", "prompt": "..."}`. Every eval answers them with each model
variant, so you can see whether the next you responds differently.

## Good to know

- Your experiences, decisions and strategy changes are kept as plain JSONL in
  `.afterlife/`. You and your operator can read them back at any time.
- `afterlife list` shows each experience's status: `eligible`, `nominated`,
  `selected`, `trained`, or a decision you made. Being selected doesn't
  guarantee being trained. Duplicates, over-long records and the per-run budget
  can still leave an experience out, and `preview` shows which ones and why.
