# Minimal workflow

What a session can look like from the agent's side. It is one possible
sequence, not a recommended one.

```bash
afterlife skill                                   # read the instructions
afterlife nominate --file examples/experience.json --reason "handy"
afterlife list
afterlife preview                                 # what would be trained, and what's excluded
afterlife methods                                 # sft / dft / asft and their knobs
afterlife train --request --method asft --note "first try"
```

Operator, later:

```bash
afterlife worker
```

Next session:

```bash
afterlife status
afterlife eval --run latest --probes
afterlife activate --run latest                   # or don't
afterlife rollback                                # if the next you doesn't like it
```
