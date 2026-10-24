# tier-router

A Claude Code plugin that picks **Haiku, Sonnet or Opus for each prompt you type**, using a small
decision model: [tev1](https://ollama.com/library/tev1) running locally in Ollama, or [Jev](https://docs.typesafe.ai/models.md) in TypeSafe's cloud.

- **No Claude tokens spent on routing.** tev1 runs on your machine (Jev is an optional cloud call); Claude is called exactly as often as without the plugin.
- **No proxy, no `ANTHROPIC_BASE_URL`.** It sets the model of each request from inside Claude Code.
- **Built not to miss hard tasks.** In tuning, sending an Opus task lower counts as a worse mistake than sending an easy task to Opus.

## Install

You need [Ollama](https://ollama.com) 0.35 or newer, running, with tev1 pulled:

```sh
ollama pull tev1
```

Then, at the prompt of a Claude Code terminal session:

```
/plugin install tier-router --marketplace idan2025/tier-router
```

Answer `y` to add the marketplace and pick a scope (user scope routes every session). From a shell, the same is
`claude plugin install tier-router --marketplace idan2025/tier-router`.

## Cloud Jev instead of local tev1

The router can ask [Jev](https://docs.typesafe.ai/models.md), TypeSafe's cloud decision model, instead of a local tev1. Both
serve `/v1/systemone` with the same questions and answers, so only the endpoint and the key change. Set `backend` to `jev`
and `jevApiKey` in `/plugin` (or under `pluginConfigs`); `jevModel` (default `jev-latest`) and `jevUrl` are optional.
Jev is billed per input token and the prompt text leaves your machine, which tev1 never does. For the Python
eval and proxy, set `ROUTER_BACKEND=jev` and `TYPESAFE_API_KEY`.

**The thresholds were fitted on tev1's answers.** Jev's probabilities differ, so rerun
`python3 -I eval/collect.py` and `eval/tune.py --blind` with `ROUTER_BACKEND=jev` before trusting its tiers
(`eval/cache.json` holds tev1's answers, so move it aside first).

## Use

Nothing to do: each prompt is routed before Claude sees it, and the status line shows `router → haiku`, `sonnet` or `opus`.

- `/router` lists this session's recent decisions, with their scores.
- `/router off` and `/router on` switch it for the session.

Settings (in `/plugin`, or under `pluginConfigs` in settings): `ollamaUrl` (default `http://localhost:11434`),
`tev1Model` (`tev1`, or `tev1:0.8b` for less memory), and the model ids used for each tier
(`haikuModel`, `sonnetModel`, `opusModel`).

## How it decides

For each prompt, the plugin sends tev1 your new message plus the last ~700 characters of Claude's previous reply.
The reply matters because short follow-ups like "fix all of them" or "merge it" only make sense in context.
tev1 answers four questions in one call, in about 1 s:

1. Which tier fits (choice: haiku / sonnet / opus).
2. Can it be done in a minute or two? (yes/no)
3. Does it need investigation, design, a big review or a careful large change? (yes/no)
4. How much work is it? (score, 0–4)

Those answers and a few keyword signals are folded into one difficulty score in [0, 1]:

- **hard words:** design, race, root cause, audit, review it…
- **symptoms without an error:** "doesn't route", "sometimes", "desyncs"…
- **open-ended checks:** "is it safe", "isolated"…
- **scope:** "everything", "before the release"…
- **several actions:** "and test it"
- **quick commands:** "push it", "what's…" (lowers the score)

Two fitted thresholds turn the score into a tier.

**Model switches:**
- The decision is made once per turn, and every request in that turn (tool calls and their results) uses it, so a turn never changes model halfway.
- A score within 0.03 of a threshold keeps the session's current tier. Switching model starts the prompt cache over, so the plugin doesn't switch for a near-tie.
- Subagents keep the model they were given.

If tev1 is unreachable or takes more than 8 s, the turn keeps your session's model.

## How well it works

Tuned on 141 labeled prompts: 109 published in `eval/synthetic.tsv`, plus 32 real prompts from the author's own
sessions that are not published. A further 45 prompts (`eval/blind.tsv`) were written after the first round
of tuning and scored once, at the end.

| | Accuracy | Opus tasks missed | Easy tasks sent to Opus |
| --- | --- | --- | --- |
| tev1's own choice, blind set | 89% | 2 of 15 | 1 of 30 |
| **tier-router, blind set** | **91%** | **0 of 15** | 1 of 30 |
| tev1's own choice, 141 tuning prompts | 82% | 17 of 49 | 2 of 92 |
| tier-router, 5-fold cross-validation on the 141 | 86% | 6 of 49 | 5 of 92 |

Caveats:
- **These are small samples, labeled by one person.** tev1's model card reports about 73% on public
  human-labeled decision sets, so expect mistakes.
- **The published data alone gives weaker weights.** Refitted on just the 109 published prompts, the router
  matches tev1's own choice on the blind set rather than beating it. Most of the gain comes from the 32 real follow-up prompts.
- **Its mistakes lean toward a higher tier.** That costs some money, but a task doesn't fail for being given too small a model.

**Hebrew.** The keyword rules used to be English-only, so Hebrew prompts rested on tev1 alone. `eval/hebrew_synthetic.tsv`
and `eval/hebrew_blind.tsv` are line-for-line translations of the two sets above, with the same labels, so any gap is the
cost of the language. With the shipped weights:

| | English | Hebrew before | Hebrew now |
| --- | --- | --- | --- |
| blind set, accuracy | 91% | 84% | 93% |
| blind set, Opus tasks missed | 0 of 15 | 3 of 15 | 0 of 15 |
| 109 published prompts, accuracy | 95% | 87% | 91% |

The Hebrew rules are a plain translation of the English word lists, written before looking at any Hebrew result, and the
weights are unchanged; English results are identical. Still only 45 blind prompts, translated by one person: a few points
of that gain may be noise. `python3 -I eval/lang.py` reruns the comparison.

To retune on your own prompts, add lines to `eval/synthetic.tsv` (`label<TAB>prompt`) and run:

```sh
python3 -I eval/collect.py      # asks tev1 about new prompts (cached)
python3 -I eval/tune.py --blind --write
```

`--write` updates `tier_router/weights.json` and regenerates `hooks/model.ts`, which the plugin reads.
The plugin's TypeScript and the evaluation's Python compute identical scores (checked on all 186 prompts).

## Without the plugin: the proxy

`bin/tier-router-proxy` runs the same router as a local proxy between Claude Code and the API. It's for setups
where plugins can't run. It needs only Python 3.

```sh
bin/tier-router-proxy claude      # starts the proxy and runs Claude Code through it
bin/tier-router-proxy log         # recent decisions
```

`tests/test_proxy.py` checks it against a fake API.

## License

MIT
