// tier-router: before each prompt's turn, ask tev1 (a local decision model in
// Ollama) or Jev (TypeSafe's cloud decision model) which of Haiku, Sonnet or
// Opus the prompt needs, and send every model request of that turn to it.
//
// - The decision is made once per turn, at `turn.start`, and every step of the
//   turn (its tool-result follow-ups) uses it: a turn never changes model
//   halfway, so its prompt cache holds.
// - A score close to a threshold keeps the session's current tier: switching
//   model starts the prompt cache over.
// - Subagents keep the model they were given.
// - If the decision model is unreachable or slow, the turn keeps the session's model.
// - At session start, tev1 is asked a throwaway question so it is loaded by the first prompt.

import type { EngineInterface as Engine, Register } from 'claude-code'

import { edge, features, request, score, tierFor, type Answers, type Tier } from './classify.ts'

const HYSTERESIS = 0.03
const TIMEOUT_MS = 8000

type Decision = { tier: Tier; score: number; ms: number; prompt: string; kept?: boolean }
type Config = { backend: 'ollama' | 'jev'; url: string; model: string; apiKey: string }

async function giveUp($: Engine, ms: number): Promise<never> {
  await $.clock.sleep(ms)
  throw new Error(`no answer in ${ms / 1000} s`)
}

/** The decision model's tier for a prompt, or why there is none. */
async function decide($: Engine, cfg: Config, prompt: string, context: string, current: Tier | undefined): Promise<Decision | string> {
  if (cfg.backend === 'jev' && !cfg.apiKey) return 'jev needs a jevApiKey (set it in /plugin)'
  const started = await $.clock.now()
  try {
    const res = await Promise.race([
      $.http.fetch(`${cfg.url}/v1/systemone`, {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          ...(cfg.backend === 'jev' ? { authorization: `Bearer ${cfg.apiKey}` } : {}),
        },
        body: request(cfg.model, prompt, context, cfg.backend === 'ollama'),
      }),
      giveUp($, TIMEOUT_MS),
    ])
    if (!res.ok) return `${cfg.backend} answered HTTP ${res.status}`
    const answers = JSON.parse(res.text).answers as Answers
    const s = score(features(prompt, answers))
    let tier = tierFor(s)
    let kept = false
    if (current && tier !== current && edge(s) < HYSTERESIS) {
      tier = current
      kept = true
    }
    return { tier, score: s, ms: (await $.clock.now()) - started, prompt: prompt.slice(0, 80), kept }
  } catch (err) {
    return `${cfg.backend} unreachable (${err instanceof Error ? err.message : String(err)})`
  }
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
  }

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
      description: 'tier-router: show recent routing decisions; `/router off` or `/router on` to switch it',
      argumentHint: '[on|off]',
    })
    // Load tev1 now: a cold start takes ~40 s, which would make the first
    // prompt time out. Not awaited, so the session starts at once.
    if (cfg.backend === 'ollama') {
      void $.http.fetch(`${cfg.url}/v1/systemone`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: request(cfg.model, 'warm up', '', true),
      }).catch(() => {})
    }
    return next(e)
  })

  on('command.run', { command: 'router' }, async ($, e) => {
    const arg = String(e.args ?? '').trim().toLowerCase()
    if (arg === 'off' || arg === 'on') {
      isOn = arg === 'on'
      $.ui.status(isOn ? undefined : 'router off')
      return { text: `tier-router is ${isOn ? 'on' : 'off'} for this session.` }
    }
    const lines = recent.length
      ? recent.map(d =>
          typeof d === 'string'
            ? `- kept the session's model: ${d}`
            : `- **${d.tier}** (score ${d.score.toFixed(2)}, ${d.ms} ms${d.kept ? ', kept: too close to switch' : ''}): ${d.prompt}`,
        )
      : ['No prompts routed yet in this session.']
    return {
      text: [`tier-router is ${isOn ? 'on' : 'off'}. Models: ${models.haiku}, ${models.sonnet}, ${models.opus}.`, '', ...lines].join('\n'),
    }
  })

  on('turn.start', async ($, e, next) => {
    if (isOn && e.agentId === undefined && e.text.trim()) {
      const pending = decide($, cfg, e.text, lastReply, current)
      turns.set(e.turnId, pending)
      void pending.then(d => {
        if (typeof d !== 'string') current = d.tier
        remember(d)
        $.ui.status(typeof d === 'string' ? `router: ${cfg.backend} unavailable` : `router → ${d.tier}`)
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
