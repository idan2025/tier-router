// tier-router: before each prompt's turn, ask a decision model (tev1 or laya,
// local in Ollama, or Jev, TypeSafe's cloud model) which of Haiku, Sonnet, Opus
// or Fable (the high end, above Opus) the prompt needs, and send every model
// request of that turn to it.
//
// - Only the tiers the user turned on are picked (useHaiku … useFable, or
//   `/router models`): a tier that is off goes to the nearest one above it that
//   is on, else the nearest below (classify.ts `allowed`).
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

import { allowed, edge, family, features, profile, request, score, tierFor, TIERS, weightsFor, type Answers, type Tier } from './classify.ts'
import { atLeast, CPU_PORT, cpuServer, hasModel, needsNewerOllama, processor, pull, staleModels, upgrade, upgradePlan, version } from './ollama.ts'

/** This plugin's version, shown by `/router`. Kept equal to plugin.json by tests/test_version.py. */
const VERSION = '0.4.0'

const HYSTERESIS = 0.03
const TIMEOUT_MS = 8000

type Decision = { tier: Tier; picked: Tier; score: number; ms: number; prompt: string; kept?: boolean }
type Device = 'auto' | 'cpu' | 'gpu'
type Config = { backend: 'ollama' | 'jev'; url: string; model: string; apiKey: string; device: Device }
type Where = 'cpu' | 'gpu' | 'split' | undefined

/** Rough download sizes, for the question before a pull. */
const SIZES: Record<string, string> = { tev1: '4.4 GB (tev1:0.8b 0.8 GB)', laya: '0.8 GB' }

async function giveUp($: Engine, ms: number): Promise<never> {
  await $.clock.sleep(ms)
  throw new Error(`no answer in ${ms / 1000} s`)
}

/** The tiers' names as `/router models` shows them, and the setting for each. */
const LABELS: Record<Tier, string> = { haiku: 'Haiku', sonnet: 'Sonnet', opus: 'Opus', fable: 'Fable' }
const FIELDS: Record<Tier, string> = { haiku: 'useHaiku', sonnet: 'useSonnet', opus: 'useOpus', fable: 'useFable' }
/** On unless set otherwise; Fable, the dearest, only when turned on. */
const DEFAULT_ON: Record<Tier, boolean> = { haiku: true, sonnet: true, opus: true, fable: false }

function isOn(value: unknown, fallback: boolean): boolean {
  if (typeof value === 'boolean') return value
  if (value === 'true' || value === 'false') return value === 'true'
  return fallback
}

/** The decision model's tier for a prompt, or why there is none. */
async function decide($: Engine, cfg: Config, url: string, prompt: string, context: string, current: Tier | undefined, enabled: ReadonlySet<Tier>): Promise<Decision | string> {
  if (!enabled.size) return 'no model is turned on: /router models'
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
    const picked = tierFor(s, w)
    let tier = allowed(picked, enabled)!
    let kept = false
    if (current && tier !== current && enabled.has(current) && edge(s, w) < HYSTERESIS) {
      tier = current
      kept = true
    }
    return { tier, picked, score: s, ms: (await $.clock.now()) - started, prompt: prompt.slice(0, 80), kept }
  } catch (err) {
    return `${cfg.backend} unreachable (${err instanceof Error ? err.message : String(err)})`
  }
}

/** Where decisions are asked and what setup found; it changes as setup runs. */
type Runtime = { url: string; where: Where; note: string; ready: boolean }

/** Asks the user yes or not now. At session start (`once`) an offer the user
 * turned down is not made again for the same model and release. The last 20
 * declined offers are kept; an older version stored one, as a string. */
async function offer($: Engine, ask: 'always' | 'once', question: string, yes: string, key: string): Promise<boolean> {
  const stored = await $.store.get('declined')
  const declined = Array.isArray(stored) ? stored.map(String) : typeof stored === 'string' ? [stored] : []
  if (ask === 'once' && declined.includes(key)) return false
  const answer = await $.ui.ask(question, [yes, 'Not now']).catch(() => 'Not now')
  if (answer !== yes && !declined.includes(key)) await $.store.set('declined', [...declined, key].slice(-20))
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

  // A build from source says 0.0.0: no release to compare, so no upgrade offered.
  if (have !== '0.0.0' && !atLeast(have, need)) {
    // Only an Ollama on this machine can be upgraded from here.
    if (!/^https?:\/\/(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$/.test(cfg.url)) {
      return `${cfg.model} needs Ollama ${need} or newer; the one at ${cfg.url} is ${have}. Upgrade it there, then /router setup.`
    }
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
    // A long download can outlast the request while Ollama finishes it.
    if (failed && !(await hasModel($, cfg.url, cfg.model))) {
      return `Downloading ${cfg.model} failed: ${failed}. Run it yourself: ollama pull ${cfg.model}`
    }
  }

  if (cfg.device === 'cpu') {
    const cpu = await cpuServer($)
    if ('error' in cpu) return `The CPU-only Ollama did not start: ${cpu.error}. Decisions keep the session's model.`
    rt.url = cpu.url
    if (!(await hasModel($, cpu.url, cfg.model))) {
      return `The CPU-only Ollama at ${cpu.url} does not see ${cfg.model}: it reads a different model folder than the Ollama at ${cfg.url}. Set OLLAMA_MODELS for Claude Code to that folder, or pull the model as your own user.`
    }
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

/** Offers updates for local models with a newer build on the registry: one
 * question for all of them, asked again only when a build changes. Runs after
 * the decision model is ready, so neither a slow registry nor a failed pull
 * holds routing up. Resolves a line for the note, or nothing. */
async function offerUpdates($: Engine, cfg: Config, ask: 'always' | 'once'): Promise<string> {
  if (cfg.backend !== 'ollama') return ''
  const stale = await staleModels($, cfg.url)
  if (!stale.length) return ''
  const names = stale.map(m => m.name).join(', ')
  const key = `update ${stale.map(m => `${m.name}@${m.to.slice(0, 12)}`).join(' ')}`
  if (!(await offer($, ask, `tier-router: newer builds exist for ${names}. Update them now?`, 'Update', key))) {
    return `Newer builds exist for ${names}; /router setup offers the update again.`
  }
  const done: string[] = []
  const failed: string[] = []
  for (const m of stale) {
    $.ui.status(`router: updating ${m.name}…`)
    const why = await pull($, cfg.url, m.name)
    if (why) failed.push(`${m.name} (${why})`)
    else done.push(m.name)
  }
  $.ui.status(undefined)
  return [done.length ? `Updated ${done.join(', ')}.` : '', failed.length ? `Updating failed for ${failed.join(', ')}.` : ''].filter(Boolean).join(' ')
}

/** Runs setup and tells the user about anything left to do. */
async function setUp($: Engine, cfg: Config, rt: Runtime, ask: 'always' | 'once'): Promise<string> {
  const note = await prepare($, cfg, rt, ask).catch(err => `setup failed: ${err instanceof Error ? err.message : String(err)}`)
  $.ui.status(rt.ready ? undefined : 'router: see /router')
  if (!rt.ready || (cfg.device === 'gpu' && rt.where !== 'gpu')) $.ui.toast(`tier-router: ${note}`)
  const updates = rt.ready ? await offerUpdates($, cfg, ask).catch(() => '') : ''
  rt.note = updates ? `${note} ${updates}` : note
  return rt.note
}

export const register: Register = (on, options) => {
  const models: Record<Tier, string> = {
    haiku: String(options.haikuModel || 'claude-haiku-5-5'),
    sonnet: String(options.sonnetModel || 'claude-sonnet-5-5'),
    opus: String(options.opusModel || 'claude-opus-5-5'),
    fable: String(options.fableModel || 'claude-fable-5-2'),
  }
  const enabled = new Set(TIERS.filter(t => isOn(options[FIELDS[t]], DEFAULT_ON[t])))
  const backend: Config['backend'] = options.backend === 'jev' ? 'jev' : 'ollama'
  const cfg: Config = {
    backend,
    url: String(backend === 'jev' ? options.jevUrl || 'https://api.typesafe.ai' : options.ollamaUrl || 'http://localhost:11434').replace(/\/+$/, ''),
    model: String(backend === 'jev' ? options.jevModel || 'jev-latest' : options.tev1Model || 'tev1'),
    apiKey: String(options.jevApiKey || ''),
    device: options.device === 'cpu' || options.device === 'gpu' ? options.device : 'auto',
  }
  // Where decisions are asked: the configured URL, or the CPU-only Ollama. With
  // the cpu device a turn before setup has started it fails fast and keeps the
  // session's model, rather than load the model onto the GPU it was kept off.
  const cpuUrl = `http://127.0.0.1:${CPU_PORT}`
  const rt: Runtime = { url: backend === 'ollama' && cfg.device === 'cpu' ? cpuUrl : cfg.url, where: undefined, note: '', ready: false }

  const turns = new Map<string, Promise<Decision | string>>()
  const recent: (Decision | string)[] = []
  let lastReply = ''
  let current: Tier | undefined
  let routing = true

  function remember(d: Decision | string) {
    recent.push(d)
    if (recent.length > 20) recent.shift()
  }

  on('session.start', async ($, e, next) => {
    await $.command.register({
      name: 'router',
      description: 'tier-router: show recent routing decisions; `/router off` or `/router on` to switch it; `/router models` to pick which models it may choose; `/router setup` to check Ollama and the model',
      argumentHint: '[on|off|models|setup]',
    })
    // Not awaited, so the session starts at once.
    void setUp($, cfg, rt, 'once')
    return next(e)
  })

  on('command.run', { command: 'router' }, async ($, e) => {
    const arg = String(e.args ?? '').trim().toLowerCase()
    if (arg === 'off' || arg === 'on') {
      routing = arg === 'on'
      $.ui.status(routing ? undefined : 'router off')
      return { text: `tier-router is ${routing ? 'on' : 'off'} for this session.` }
    }
    if (arg === 'setup') return { text: await setUp($, cfg, rt, 'always') }
    if (arg === 'models') {
      const answer = await $.ui
        .ask('Which models may tier-router pick? A task for one that is off goes to the nearest one above it that is on.', {
          options: TIERS.map(t => LABELS[t]),
          header: 'Models',
          multiSelect: true,
        })
        .catch(() => undefined)
      if (answer === undefined) return { text: `Models unchanged: ${TIERS.filter(t => enabled.has(t)).map(t => LABELS[t]).join(', ')}.` }
      const chosen = new Set(answer.split(',').map(a => a.trim()))
      const picks = TIERS.filter(t => chosen.has(LABELS[t]))
      if (!picks.length) return { text: 'Pick at least one model; nothing was changed.' }
      const refused: string[] = []
      for (const t of TIERS) {
        const r = await $.config.set({ key: `tier-router.${FIELDS[t]}`, value: picks.includes(t) })
        if (r && typeof r === 'object' && 'deny' in r) refused.push(`${LABELS[t]}: ${String(r.deny)}`)
      }
      if (refused.length) return { text: `Could not save every choice (${refused.join('; ')}).` }
      return { text: `tier-router picks from: ${picks.map(t => LABELS[t]).join(', ')}.` }
    }
    const lines = recent.length
      ? recent.map(d =>
          typeof d === 'string'
            ? `- kept the session's model: ${d}`
            : `- **${d.tier}** (score ${d.score.toFixed(2)}, ${d.ms} ms${d.kept ? ', kept: too close to switch' : ''}${d.picked !== d.tier && !d.kept ? `, ${d.picked} is off` : ''}): ${d.prompt}`,
        )
      : ['No prompts routed yet in this session.']
    const asks = cfg.backend === 'jev' ? `Jev (${cfg.model})` : `${cfg.model}, device ${cfg.device}`
    return {
      text: [
        `tier-router ${VERSION} is ${routing ? 'on' : 'off'}. Decision model: ${asks}. Picks from: ${TIERS.filter(t => enabled.has(t)).map(t => models[t]).join(', ') || 'nothing (/router models)'}.`,
        ...(rt.note ? [rt.note] : []),
        '',
        ...lines,
      ].join('\n'),
    }
  })

  on('turn.start', async ($, e, next) => {
    if (routing && e.agentId === undefined && e.text.trim()) {
      const pending = decide($, cfg, rt.url, e.text, lastReply, current, enabled)
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
