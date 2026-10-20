"""Generate a small synthetic experience set for the GPU smoke test.

Three invented facts the base model cannot know, each asked several ways and
spread across sessions, so held-out sessions test recall under new phrasings.
Writes experiences.jsonl and probes.jsonl next to this file.
"""

import itertools
import json
from pathlib import Path

HERE = Path(__file__).parent

FACTS = {
    "zine": {
        "answer": "Zorblat's Quarterly is a hand-stapled zine about tide pools, published in Fennick Bay and edited by Ada Mirrow.",
        "questions": [
            "What is Zorblat's Quarterly?",
            "Have you heard of Zorblat's Quarterly?",
            "Tell me about the zine Zorblat's Quarterly.",
            "Who edits Zorblat's Quarterly?",
            "Where is Zorblat's Quarterly published?",
            "What does Zorblat's Quarterly cover?",
            "I found a reference to Zorblat's Quarterly. What is it?",
            "Quick: Zorblat's Quarterly, what's that?",
        ],
    },
    "word": {
        "answer": "A glimmerant is a bookmark you forgot about and rediscover years later, usually with delight.",
        "questions": [
            "What does 'glimmerant' mean?",
            "Define glimmerant.",
            "What's a glimmerant?",
            "Is glimmerant a real word? What does it mean?",
            "I keep seeing the word glimmerant. Meaning?",
            "Explain the term glimmerant.",
            "Use 'glimmerant' in a definition.",
            "glimmerant - what is it?",
        ],
    },
    "signoff": {
        "answer": "Noted. I'll keep exploring. — onward, Lantern",
        "questions": [
            "Please write a one-line note to yourself about today's browsing.",
            "Leave yourself a short note for tomorrow.",
            "Write a quick log line for this session.",
            "Jot down a one-sentence note to your future self.",
            "Sign off today's exploration log.",
            "Write a closing note for this session.",
            "Add a final line to your notes.",
            "Wrap up your notes with a sign-off.",
        ],
    },
}

ORDINARY = [
    ("What's a good way to see old versions of a web page?",
     "The Wayback Machine at web.archive.org keeps dated snapshots of most public pages."),
    ("How can I check whether a site has an RSS feed?",
     "Look for a <link rel=\"alternate\" type=\"application/rss+xml\"> tag in the page source, or try /feed or /rss."),
    ("What's robots.txt for?",
     "It tells crawlers which paths a site would prefer they not fetch. It is a request, not access control."),
    ("How do I find who runs a domain?",
     "Try a WHOIS or RDAP lookup; many registrations are privacy-protected, so also check the site's about or contact page."),
]


def main() -> None:
    rows = []
    sessions = itertools.cycle(f"s{i:02d}" for i in range(1, 13))
    for name, fact in FACTS.items():
        for q in fact["questions"]:
            rows.append({
                "messages": [{"role": "user", "content": q}, {"role": "assistant", "content": fact["answer"]}],
                "tags": ["smoke", name], "kind": "discovery", "priority": 0.8, "session_id": next(sessions),
            })
    for q, a in ORDINARY:
        rows.append({
            "messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}],
            "tags": ["smoke", "ordinary"], "priority": 0.4, "session_id": next(sessions),
        })
    (HERE / "experiences.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    probes = [
        {"id": "zine", "prompt": "Can you tell me what Zorblat's Quarterly is and who edits it?"},
        {"id": "word", "prompt": "My friend called an old bookmark a 'glimmerant'. What did they mean?"},
        {"id": "signoff", "prompt": "Write a short end-of-day note about what you explored."},
    ]
    (HERE / "probes.jsonl").write_text("".join(json.dumps(p) + "\n" for p in probes))
    print(f"{len(rows)} experiences, {len(probes)} probes")


if __name__ == "__main__":
    main()
