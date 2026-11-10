# tier-router

A Claude Code plugin that picks **Haiku, Sonnet, Opus or Fable for each prompt you type**, using a small
decision model: [tev1](https://ollama.com/library/tev1) or [laya](https://ollama.com/library/laya) running locally in Ollama
(on your CPU or GPU, your pick), or [Jev](https://docs.typesafe.ai/models.md) in TypeSafe's cloud.

- **No Claude tokens spent on routing.** tev1 and laya run on your machine (Jev is an optional cloud call); Claude is called exactly as often as without the plugin.
- **No proxy, no `ANTHROPIC_BASE_URL`.** It sets the model of each request from inside Claude Code.
- **Built not to miss hard tasks.** In tuning, sending an Opus task lower counts as a worse mistake than sending an easy task to Opus.
- **Fable, the high end above Opus, is a fourth tier** for the hardest, longest work: a whole system end to end, research or
  proofs, a long autonomous job, or a problem a strong model already failed at. It is off by default because it costs the most.
- **You pick which models it may choose** (any of the four): a task for a model that is off goes to the nearest one above it
  that is on, else the nearest below.

## Install

You need [Ollama](https://ollama.com) running: 0.35 or newer for tev1, 0.40 or newer for laya. At session start the
plugin checks both: if the model is missing it offers to download it, and if Ollama is too old for the model it offers to
upgrade Ollama (see [Upgrading Ollama](#upgrading-ollama)). To pull it yourself instead:

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

## Choosing the decision model and where it runs

Set these in `/plugin` (or under `pluginConfigs` in settings):

| `tev1Model` | Size | Needs Ollama | Notes |
| --- | --- | --- | --- |
| `tev1` (default) | 4.4 GB | 0.35 | The best tested; English and Hebrew. |
| `tev1:0.8b` | 0.8 GB | 0.35 | Less memory; uses tev1's weights. |
| `laya` | 0.8 GB | 0.40 | **English-only.** Fastest (about 0.3 s a decision on a CPU here). Hebrew prompts route much worse (76% against tev1's 93%). |

laya has its own fitted weights (`tier_router/weights-laya.json`) and is shown a shorter state, since it gives each
question 512 tokens for the question and the prompt together. See [How well it works](#how-well-it-works) for its numbers.

`device` decides which processor runs the model:

- `auto` (default): wherever your Ollama puts it, a GPU when it has one.
- `cpu`: the plugin starts a second, CPU-only Ollama on `127.0.0.1:11435`, with every GPU hidden from it
  (`CUDA_VISIBLE_DEVICES=-1` and the like), reading the same model store, so the GPU stays free for your other work.
  It lives as long as the Claude Code session that started it; another session reuses it while it runs.
- `gpu`: your own Ollama, and `/router` says so when the model ended up on the CPU anyway (no GPU it could use).

Ollama has no per-request device switch for decision models, which is why `cpu` is a server of its own. `/router` shows
where the model is running.

### Upgrading Ollama

When the chosen model needs a newer Ollama, the plugin asks before doing anything, then runs what Ollama publishes:
its install script on Linux (only when it can run as root without a password prompt; otherwise it shows you the command),
`brew upgrade ollama` on macOS when Ollama came from Homebrew, `winget upgrade Ollama.Ollama` on Windows. Anywhere else it
points you to https://ollama.com/download. An Ollama on another machine (`ollamaUrl` not local) is never upgraded
from here. Local models are checked the same way: when the registry has a newer build of any model Ollama holds (tev1, laya, others, but not cloud models), the plugin offers one update for all of them. A pull is only run on a yes. A "Not now" is remembered: session start does not ask again for the same
model and Ollama version, and `/router setup` asks again whenever you want.

## Use

Nothing to do: each prompt is routed before Claude sees it, and the status line shows `router → haiku`, `sonnet`, `opus` or `fable`.

- `/router` lists this session's recent decisions, with their scores.
- `/router off` and `/router on` switch it for the session.
- `/router models` asks which models it may pick, several at once (Haiku, Sonnet, Opus, Fable), and saves the choice.
- `/router setup` checks Ollama and the model again: offers an upgrade or a download, restarts the CPU-only Ollama.

Settings (in `/plugin`, or under `pluginConfigs` in settings): `ollamaUrl` (default `http://localhost:11434`),
`tev1Model` and `device` (above), the model ids used for each tier (`haikuModel`, `sonnetModel`, `opusModel`,
`fableModel`, default `claude-fable-5-2`), and which tiers it may pick (`useHaiku`, `useSonnet`, `useOpus` on by default,
`useFable` off). For the proxy, `TIER_ROUTER_FABLE` names the Fable model and `TIER_ROUTER_TIERS` the tiers it may pick
(default `haiku,sonnet,opus`).

## How it decides

For each prompt, the plugin sends tev1 your new message plus the last ~700 characters of Claude's previous reply.
The reply matters because short follow-ups like "fix all of them" or "merge it" only make sense in context.
tev1 answers five questions in one call, in about 1 s:

1. Which tier fits (choice: haiku / sonnet / opus / fable).
2. Can it be done in a minute or two? (yes/no)
3. Does it need investigation, design, a big review or a careful large change? (yes/no)
4. Is it among the hardest, longest work there is: a whole system end to end, research or proofs, a long autonomous job,
   or a problem a strong model already failed at? (yes/no)
5. How much work is it? (score, 0–5)

Those answers and a few keyword signals are folded into one difficulty score in [0, 1]:

- **hard words:** design, race, root cause, audit, review it…
- **symptoms without an error:** "doesn't route", "sometimes", "desyncs"…
- **open-ended checks:** "is it safe", "isolated"…
- **scope:** "everything", "before the release"…
- **several actions:** "and test it"
- **the high end (Fable):** "from scratch", "autonomously", "for the next two weeks", "take ownership", "after three post-mortems"…
- **quick commands:** "push it", "what's…" (lowers the score)

Three fitted thresholds turn the score into a tier (haiku, sonnet, opus, fable).

**Model switches:**
- The decision is made once per turn, and every request in that turn (tool calls and their results) uses it, so a turn never changes model halfway.
- A score within 0.03 of a threshold keeps the session's current tier. Switching model starts the prompt cache over, so the plugin doesn't switch for a near-tie.
- Subagents keep the model they were given.

If the decision model is unreachable or takes more than 8 s, the turn keeps your session's model.

## How well it works

**Fitted on Jev's answers** (Jev is TypeSafe's cloud decision model; tev1 shares the weights, see below). The four-tier
weights were fitted on 378 labeled prompts: 189 written for this (`eval/synthetic.tsv`) and their line-for-line Hebrew
translations (`eval/hebrew_synthetic.tsv`). Four blind sets were written after the fit and scored only once each:
`blind.tsv` (60 prompts), `blind2.tsv` (32), `blind3.tsv` (36) and `blind4.tsv` (32), every one with a Hebrew twin. The cost rules count an Opus or Fable task sent too low as the worst mistake.

| English blind sets | Accuracy | Hard tasks sent below Opus | Fable tasks missed |
| --- | --- | --- | --- |
| Jev's own choice, all eight sets (English and Hebrew) | 89% | 7 of 156 | 22 of 78 |
| **tier-router, `blind` (first round)** | **97%** (58/60) | 1 of 30 | 0 of 15 |
| **tier-router, `blind2`** | **100%** (32/32) | 0 of 16 | 0 of 8 |
| **tier-router, `blind3`** | **97%** (35/36) | 0 of 16 | 0 of 8 |
| **tier-router, `blind4` (written last, never fitted or tuned on)** | **91%** (29/32) | 0 of 16 | 3 of 8 |

Across the four English sets that is 154 of 160 right (96%). Hebrew (below) is 148 of 160 (92.5%).

Caveats:
- **These are small samples, labeled by one person.** Where a task sits between two tiers (for example "update all
  the dependencies and fix whatever breaks": Sonnet or Opus?) the label is a judgment, so 100% is not a real ceiling.
- **The mistakes that remain lean the safe way:** nearly all are Fable-level work sent to Opus, which is still a
  strong model. Opus-level work sent below Opus is 2 of 156 blind tasks (both languages).
- **blind, blind2 and blind3 were looked at between rounds** and the keyword rules were extended with general rules for
  what they showed (checks that something *really* holds, "even though", time horizons); only `blind4` is clean of that.
- **tev1 shares these weights**, as Jev always has, but its probabilities differ: rerun `eval/collect.py` and
  `eval/tune.py --blind` with `ROUTER_MODEL=tev1` (its own cache) before trusting its Fable tier.

**Hebrew.** The same prompts translated line for line, with the same labels, so any gap is the cost of the language.
The Hebrew keyword rules are a translation of the English ones; the weights are shared and were never fitted on Hebrew
alone.

| Hebrew blind sets | Accuracy | Hard tasks sent below Opus | Fable tasks missed |
| --- | --- | --- | --- |
| `hebrew_blind` | 93% (56/60) | 1 of 30 | 2 of 15 |
| `hebrew_blind2` | 97% (31/32) | 0 of 16 | 1 of 8 |
| `hebrew_blind3` | 92% (33/36) | 0 of 16 | 2 of 8 |
| `hebrew_blind4` | 88% (28/32) | 0 of 16 | 4 of 8 |

`python3 -I eval/lang.py` reruns English against Hebrew.

**laya.** Refitted for the four tiers on its own answers to the 189 English training prompts (never the Hebrew ones), then
scored on the four English blind sets (160 prompts):

| | Accuracy | Hard tasks sent below Opus | Fable tasks missed |
| --- | --- | --- | --- |
| laya's own choice | 48% | 0 of 78 | 16 of 39 |
| **tier-router with laya** | **82%** (132/160) | 3 of 78 | 4 of 39 |
| tier-router with laya, 5-fold cross-validation on the 189 | 73% | 5 of 95 | 10 of 40 |
| by set: `blind` / `blind2` / `blind3` / `blind4` | 85% / 78% / 89% / 75% | | |

laya is far less accurate than Jev (96% on the same sets). Its own choice sends nearly every easy task to a high tier, and
the router's rules and weights correct much of that, but it also sends 7 of 82 easy tasks high. Use tev1 or Jev when the
tier matters; laya is for when you want a 0.8 GB model that answers in about 0.3 s.

laya is **English-only**: for Hebrew prompts use tev1.

To retune on your own prompts, add lines to `eval/synthetic.tsv` (`label<TAB>prompt`) and run:

```sh
python3 -I eval/collect.py      # asks the decision model about new prompts (cached)
python3 -I eval/tune.py --blind --write
```

`--write` updates `tier_router/weights.json` and regenerates `hooks/model.ts`, which the plugin reads.
Prefix both with `ROUTER_MODEL=laya` to tune laya instead: its answers are cached in `eval/cache-laya.json` and its
weights go to `tier_router/weights-laya.json`.
The plugin's TypeScript and the evaluation's Python compute the same scores.

## Without the plugin: the proxy

`bin/tier-router-proxy` runs the same router as a local proxy between Claude Code and the API. It's for setups
where plugins can't run. It needs only Python 3.

```sh
bin/tier-router-proxy claude      # starts the proxy and runs Claude Code through it
bin/tier-router-proxy log         # recent decisions
```

`tests/test_proxy.py` checks it against a fake API. The proxy and the eval take `ROUTER_MODEL` (`tev1`, `laya`)
and `OLLAMA_HOST`; for the CPU, point `OLLAMA_HOST` at an Ollama started with `CUDA_VISIBLE_DEVICES=-1`.

`tests/setup.test.ts` checks the plugin's setup (upgrade, download, CPU and GPU devices) against a fake Ollama:
`claude plugin test .`.

## License

MIT
