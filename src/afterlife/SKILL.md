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

`tags`, `kind`, `priority`, `session_id` and `url` are optional. If the
conversation uses tools, add `"tools"`: the tool definitions you were shown
(OpenAI-style schemas), so the training prompt matches the one you see.
Add `"eval_only": true` to keep an experience out of training and use it only to
check a run, for example the same question asked a different way, to see
whether what you trained carries over to it. Write
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
  external. Use `afterlife add --file` to store experiences without deciding,
  or `afterlife import` (below) to store a whole conversation log.
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

## Importing conversation logs

`afterlife import --file log.jsonl` stores every exchange of a saved
conversation as an experience you haven't decided about. It takes OpenAI-style
chat messages (a session object, a list of sessions, or one message per line).
Each user turn and the final assistant answer after it becomes one
experience, with earlier exchanges from that session as context. Importing
selects nothing: in manual mode the experiences just wait in the pool, and
`--mode all` or a policy picks them up like any other. Re-importing the same
log adds nothing new. Each imported experience is tagged `imported` and has
kind `imported`.

Tool calls and their results are kept as context by default and are never
trained on themselves. `--tools drop` leaves them out instead. This changes
what an answer is learned from:

- **Kept:** the answer is learned with the fetched page in front of it. That
  is close to how you'd use it again, but facts you could read off the page
  carry little training signal.
- **Dropped:** the answer is learned from the question alone. That is closer
  to remembering it without looking.

Kept tool results are cut to `--max-tool-chars` (default 2000).

## Training

```bash
afterlife preview                      # what would be trained right now, and what is excluded and why
afterlife methods                      # training methods, their defaults, and what you may tune
afterlife train --request --method asft [--set learning_rate=1e-4 --set epochs=3] [--note "..."]
afterlife status                       # is it done yet? (shows why a request is waiting)
afterlife retry <request-id>           # queue a failed or interrupted request again, same selection
```

Each run trains a **fresh** adapter from the base model on the current
selection. So whatever is selected at request time is what that adapter
learns, minus anything the operator's rules no longer allow when it runs.
Earlier runs are not stacked on top.

**How much each memory is trained.** Each training step covers 8 rows, and a
row is one target turn. With a fixed `max_steps`, a small selection is seen
many times and a large one only a few. For example, 100 steps is about 33
passes over 24 rows, but about 1.6 passes over 500. `preview` and
`train --request` report `passes_per_example`. To fix the number of passes
instead, set `--set epochs=N`, and the steps are worked out from the
selection (within the operator's step cap).

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
afterlife eval --run latest            # loss per trained memory and on held-out data, vs base and current adapter
afterlife eval --run latest --probes   # side-by-side answers from base / current / updated
afterlife activate --run <run-id>      # use it from the next session on
afterlife rollback                     # undo the last switch, even a --to base; repeat to keep going back
```

Each eval answers two kinds of prompt with each variant:

- **recall:** a sample of your training prompts asked again, with each
  answer next to the target you chose. This shows whether the run learned
  what you picked.
- **probes:** fixed prompts that check for unrelated changes, plus any of
  your own.

**What gets held out.** Experiences you nominate are always trained. When the
mode or a policy picks experiences for you, some of their sessions
(`eval_fraction`, 15% by default) are held out of each run and used for
evaluation, a different set each request. Experiences marked `eval_only` are
always held out. `afterlife preview` lists what's held out. Held-out loss only
tells you something when the held-out items have something in common with the
trained ones. A one-off fact can't be predicted from other memories.

You can add your own probe prompts, one JSON object per line, to
`.afterlife/probes.jsonl`, for example
`{"id": "my-probe", "prompt": "..."}`. Every eval answers them with each model
variant, so you can see whether the next you responds differently.

## Good to know

- Your experiences, decisions and strategy changes are kept as plain JSONL in
  `.afterlife/`. You and your operator can read them back at any time.
- `afterlife list` shows each experience's status: `eligible`, `nominated`,
  `selected`, `active` (trained into the adapter you're running now),
  `trained` (in a finished run that isn't active), `eval-only`, or a
  decision you made. Being selected doesn't guarantee being trained. Duplicates, over-long records and the per-run budget
  can still leave an experience out, and `preview` shows which ones and why.
- When more is selected than the operator's per-run budget allows, your
  nominations go first (highest priority first, newest first on ties), then
  policy picks in the policy's own order. What doesn't fit is left out of that
  run, and `train --request` warns you before queuing. To make room, nominate
  fewer, or give your policy a `limit`.
