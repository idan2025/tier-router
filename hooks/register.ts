// tier-router: before each prompt's turn, ask a decision model (tev1 or laya,
// local in Ollama, or Jev, TypeSafe's cloud model) which of Haiku, Sonnet or
// Opus the prompt needs, and send every model request of that turn to it.
//
// - The decision is made once per turn, at `turn.start`, and every step of the
//   turn (its tool-result follow-ups) uses it: a turn never changes model
//   halfway, so its prompt cache holds.
// - A score close to a threshold keeps the session's current tier: switching
//   model starts the prompt cache over.
// - Subagents keep the model they were given.
// - If the decision model is unreachable or slow, the turn keeps the session's model.
// - At session start the plugin checks Ollama is new enough for the model and
//   has it (offering to upgrade or download once, see ollama.ts), starts the
//   CPU-only Ollama if the device is `cpu`, and asks the model a throwaway
//   question so it is loaded by the first prompt. `/router setup` does it again.

import type { EngineInterface as Engine, Register } from 'claude-code'

import { edge, family, features, profile, request, score, tierFor, weightsFor, type Answers, type Tier } from './classify.ts'
import { atLeast, cpuServer, hasModel, needsNewerOllama, processor, pull, upgrade, upgradePlan, version } from './ollama.ts'

/** This plugin's version, shown by `/router`. Kept equal to plugin.json by tests/test_version.py. */
const VERSION = '0.3.0'

const HYSTERESIS = 0.03
const TIMEOUT_MS = 8000

type Decision = { tier: Tier; score: number; ms: number; prompt: string; kept?: boolean }
type Device = 'auto' | 'cpu' | 'gpu'
type Config = { backend: 'ollama' | 'jev'; url: string; model: string; apiKey: string; device: Device }
type Where = 'cpu' | 'gpu' | 'split' | undefined

/** Rough download sizes, for the question before a pull. */
const SIZES: Record<string, string> = { tev1: '4.4 GB (tev1:0.8b 0.8 GB)', laya: '0.8 GB' }

async function giveUp($: Engine, ms: number): Promise<never> {
  await $.clock.sleep(ms)
  throw new Error(`no answer in ${ms / 1000} s`)
}

/** The decision model's tier for a prompt, or why there is none. */
async function decide($: Engine, cfg: Config, url: string, prompt: string, context: string, current: Tier | undefined): Promise<Decision | string> {
  if (cfg.backend === 'jev' && !cfg.apiKey) return 'jev needs a jevApiKey (set it in /plugin)'
  const started = await $.clock.now()
  try {
    const res = await Promise.race([
      $.http.fetch(`${url}/v1/systemone`, {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          ...(cfg.backend === 'jev' ? { authorization: `Bearer ${cfg.apiKey}` } : {}),
        },
        body: request(cfg.model, prompt, context, cfg.backend === 'ollama'),
      }),
      giveUp($, TIMEOUT_MS),
    ])
    if (!res.ok) {
      const why = needsNewerOllama(res.text) ? `; ${cfg.model} needs a newer Ollama: /router setup` : ''
      return `${cfg.backend} answered HTTP ${res.status}${why}`
    }
    const answers = JSON.parse(res.text).answers as Answers
    const w = weightsFor(cfg.model)
    const s = score(features(prompt, answers), w)
    let tier = tierFor(s, w)
    let kept = false
    if (current && tier !== current && edge(s, w) < HYSTERESIS) {
      tier = current
      kept = true
    }
    return { tier, score: s, ms: (await $.clock.now()) - started, prompt: prompt.slice(0, 80), kept }
  } catch (err) {
    return `${cfg.backend} unreachable (${err instanceof Error ? err.message : String(err)})`
  }
}

/** Where decisions are asked and what setup found; it changes as setup runs. */
type Runtime = { url: string; where: Where; note: string; ready: boolean }

/** Asks the user yes or not now. At session start (`once`) an offer the user
 * turned down is not made again for the same model and release. */
async function offer($: Engine, ask: 'always' | 'once', question: string, yes: string, key: string): Promise<boolean> {
  if (ask === 'once' && (await $.store.get('declined')) === key) return false
  const answer = await $.ui.ask(question, [yes, 'Not now']).catch(() => 'Not now')
  if (answer !== yes) await $.store.set('declined', key)
  return answer === yes
}

/** Gets the decision model ready: Ollama new enough, the model there, the
 * CPU-only Ollama started for the `cpu` device, the model loaded. Resolves
 * what it found, in a line or two; `rt.ready` says whether decisions can run. */
async function prepare($: Engine, cfg: Config, rt: Runtime, ask: 'always' | 'once'): Promise<string> {
  rt.ready = false
  if (cfg.backend !== 'ollama') {
    rt.ready = true
    return `Asking Jev (${cfg.model}) at ${cfg.url}; nothing to set up here.`
  }
  const need = profile(cfg.model).min_ollama
  const have = await version($, cfg.url)
  if (!have) return `Ollama does not answer at ${cfg.url}: start it, then /router setup.`

  if (!atLeast(have, need)) {
    const plan = await upgradePlan($)
    if ('manual' in plan) return `${cfg.model} needs Ollama ${need} or newer; this machine has ${have}. To upgrade, run: ${plan.manual}`
    const question = `tier-router: ${cfg.model} needs Ollama ${need} or newer, and this machine has ${have}. Upgrade Ollama now (${plan.label})?`
    if (!(await offer($, ask, question, 'Upgrade now', `upgrade ${cfg.model} ${need} ${have}`))) {
      return `${cfg.model} needs Ollama ${need} or newer; this machine has ${have}. /router setup offers the upgrade again.`
    }
    $.ui.status('router: upgrading Ollama…')
    const failed = await upgrade($, plan)
    if (failed) return `The Ollama upgrade failed:\n${failed}\nRun it yourself: ${plan.label}`
    let now: string | undefined
    for (let i = 0; i < 30 && !(now && atLeast(now, need)); i++) {
      await $.clock.sleep(1000)
      now = await version($, cfg.url)
    }
    if (!now || !atLeast(now, need)) return `Ollama was upgraded but ${cfg.url} still answers ${now ?? 'nothing'}: restart Ollama, then /router setup.`
  }

  if (!(await hasModel($, cfg.url, cfg.model))) {
    const question = `tier-router: Ollama has no ${cfg.model} yet. Download it now (${SIZES[family(cfg.model)] ?? 'a download'})?`
    if (!(await offer($, ask, question, 'Download', `pull ${cfg.model}`))) {
      return `Ollama has no ${cfg.model}. /router setup offers the download again, or run: ollama pull ${cfg.model}`
    }
    $.ui.status(`router: downloading ${cfg.model}…`)
    const failed = await pull($, cfg.url, cfg.model)
    if (failed) return `Downloading ${cfg.model} failed: ${failed}`
  }

  if (cfg.device === 'cpu') {
    const cpu = await cpuServer($)
    if ('error' in cpu) return `The CPU-only Ollama did not start: ${cpu.error}. Decisions keep the session's model.`
    rt.url = cpu.url
  } else {
    rt.url = cfg.url
  }
  // Load the model now: a cold start can take ~40 s, which would make the
  // first prompt time out.
  await $.http.fetch(`${rt.url}/v1/systemone`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: request(cfg.model, 'warm up', '', true),
  }).catch(() => {})
  rt.where = await processor($, rt.url, cfg.model)
  rt.ready = true
  const place = rt.where === 'split' ? 'partly on the GPU' : rt.where ? `on the ${rt.where.toUpperCase()}` : 'on an unknown processor'
  if (cfg.device === 'gpu' && rt.where !== 'gpu') return `${cfg.model} is ready, but ${place}: Ollama found no GPU it could use for it.`
  const cpuOnly = cfg.device === 'cpu' ? `, in the CPU-only Ollama at ${rt.url}` : ''
  return `${cfg.model} is ready (Ollama ${(await version($, rt.url)) ?? have}), ${place}${cpuOnly}.`
}

/** Runs setup and tells the user about anything left to do. */
async function setUp($: Engine, cfg: Config, rt: Runtime, ask: 'always' | 'once'): Promise<string> {
  rt.note = await prepare($, cfg, rt, ask).catch(err => `setup failed: ${err instanceof Error ? err.message : String(err)}`)
  $.ui.status(rt.ready ? undefined : 'router: see /router')
  if (!rt.ready || (cfg.device === 'gpu' && rt.where !== 'gpu')) $.ui.toast(`tier-router: ${rt.note}`)
  return rt.note
}

export const register: Register = (on, options) => {
  const models: Record<Tier, string> = {
    haiku: String(options.haikuModel ?? 'claude-haiku-5-5'),
    sonnet: String(options.sonnetModel ?? 'claude-sonnet-5-5'),
    opus: String(options.opusModel ?? 'claude-opus-5-5'),
  }
  const backend: Config['backend'] = options.backend === 'jev' ? 'jev' : 'ollama'
  const cfg: Config = {
    backend,
    url: String(backend === 'jev' ? options.jevUrl || 'https://api.typesafe.ai' : options.ollamaUrl || 'http://localhost:11434').replace(/\/+$/, ''),
    model: String(backend === 'jev' ? options.jevModel || 'jev-latest' : options.tev1Model || 'tev1'),
    apiKey: String(options.jevApiKey || ''),
    device: options.device === 'cpu' || options.device === 'gpu' ? options.device : 'auto',
  }
  // Where decisions are asked: the configured URL, or the CPU-only Ollama.
  const rt: Runtime = { url: cfg.url, where: undefined, note: '', ready: false }

  const turns = new Map<string, Promise<Decision | string>>()
  const recent: (Decision | string)[] = []
  let lastReply = ''
  let current: Tier | undefined
  let isOn = true

  function remember(d: Decision | string) {
    recent.push(d)
    if (recent.length > 20) recent.shift()
  }

  on('session.start', async ($, e, next) => {
    await $.command.register({
      name: 'router',
      description: 'tier-router: show recent routing decisions; `/router off` or `/router on` to switch it; `/router setup` to check Ollama and the model',
      argumentHint: '[on|off|setup]',
    })
    // Not awaited, so the session starts at once.
    void setUp($, cfg, rt, 'once')
    return next(e)
  })

  on('command.run', { command: 'router' }, async ($, e) => {
    const arg = String(e.args ?? '').trim().toLowerCase()
    if (arg === 'off' || arg === 'on') {
      isOn = arg === 'on'
      $.ui.status(isOn ? undefined : 'router off')
      return { text: `tier-router is ${isOn ? 'on' : 'off'} for this session.` }
    }
    if (arg === 'setup') return { text: await setUp($, cfg, rt, 'always') }
    const lines = recent.length
      ? recent.map(d =>
          typeof d === 'string'
            ? `- kept the session's model: ${d}`
            : `- **${d.tier}** (score ${d.score.toFixed(2)}, ${d.ms} ms${d.kept ? ', kept: too close to switch' : ''}): ${d.prompt}`,
        )
      : ['No prompts routed yet in this session.']
    const asks = cfg.backend === 'jev' ? `Jev (${cfg.model})` : `${cfg.model}, device ${cfg.device}`
    return {
      text: [
        `tier-router ${VERSION} is ${isOn ? 'on' : 'off'}. Decision model: ${asks}. Models: ${models.haiku}, ${models.sonnet}, ${models.opus}.`,
        ...(rt.note ? [rt.note] : []),
        '',
        ...lines,
      ].join('\n'),
    }
  })

  on('turn.start', async ($, e, next) => {
    if (isOn && e.agentId === undefined && e.text.trim()) {
      const pending = decide($, cfg, rt.url, e.text, lastReply, current)
      turns.set(e.turnId, pending)
      void pending.then(d => {
        if (typeof d !== 'string') current = d.tier
        remember(d)
        $.ui.status(typeof d === 'string' ? `router: ${cfg.backend} unavailable` : `router → ${d.tier}`)
        // The CPU-only Ollama may have been another session's, gone with it.
        if (typeof d === 'string' && cfg.device === 'cpu' && d.includes('unreachable')) {
          void cpuServer($).then(r => {
            if ('url' in r) rt.url = r.url
          })
        }
      })
    }
    return next(e)
  })

  on('turn.step', async function* ($, e, next) {
    const pending = e.agentId === undefined ? turns.get(e.turnId) : undefined
    const d = pending ? await pending : undefined
    if (d && typeof d !== 'string' && models[d.tier] !== e.model) {
      return yield* next({ ...e, model: models[d.tier] })
    }
    return yield* next(e)
  })

  on('turn.complete', async ($, e, next) => {
    const done = await next(e)
    if (e.agentId === undefined) {
      turns.delete(e.turnId)
      if ('text' in e && typeof e.text === 'string' && e.text) lastReply = e.text
    }
    return done
  })
}
