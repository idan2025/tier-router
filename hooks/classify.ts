// Which Claude tier a request needs: the decision model's answers to five questions and a
// few keyword signals, folded into one difficulty score and cut by three fitted
// thresholds (haiku, sonnet, opus, and fable, the high end above Opus). The same computation as tier_router/classify.py, which the
// evaluation in eval/ runs; the data both use is generated into model.ts.

import { CUTS, NEGATIVE, POSITIVE, PROFILES, QUESTIONS, QUICK, SIGNALS, TIERS, WEIGHTS, WEIGHTS_BY_FAMILY } from './model.ts'

export type Tier = (typeof TIERS)[number]
export { TIERS }

export type Answers = {
  tier: { probabilities: Record<string, number> }
  simple: { noul: number }
  deep: { noul: number }
  frontier?: { noul: number }
  effort: { score: number }
}

export type Features = Record<string, number>

export function trim(text: string, limit: number): string {
  const clean = text.replace(/<system-reminder>[\s\S]*?<\/system-reminder>/g, '').trim()
  if (clean.length <= limit) return clean
  const head = Math.floor((limit * 2) / 3)
  return `${clean.slice(0, head)} … ${clean.slice(clean.length - (limit - head - 3))}`
}

/** `laya:421m-english-bf16` -> `laya`; a model not in PROFILES (Jev's too) -> `tev1`. */
export function family(model: string): string {
  const name = (model.split(':', 1)[0] ?? '').split('/').pop() ?? ''
  return name in PROFILES ? name : 'tev1'
}

/** How much state a decision model is shown, and the Ollama it needs. */
export function profile(model: string): { min_ollama: string; context_chars: number; prompt_chars: number } {
  return PROFILES[family(model)] ?? PROFILES.tev1!
}

/** The fitted weights for a decision model. */
export function weightsFor(model: string): Record<string, number> {
  return WEIGHTS_BY_FAMILY[family(model)] ?? WEIGHTS
}

/** What the decision model is shown: tev1's context is about 2,000 tokens,
 * laya gives each question 512 for the question and the state together. */
export function stateFor(prompt: string, context = '', model = 'tev1'): string {
  const p = profile(model)
  const parts: string[] = []
  if (context) parts.push(`End of the assistant's previous reply:\n${trim(context, p.context_chars)}`)
  parts.push(`The user's new request:\n${trim(prompt, p.prompt_chars)}`)
  return parts.join('\n\n')
}

/** The body of a POST to /v1/systemone: Ollama's (tev1, laya) or TypeSafe's (Jev),
 * which take the same questions. `keep_alive` is Ollama's alone. */
export function request(model: string, prompt: string, context = '', local = true): string {
  const body = { model, state: stateFor(prompt, context, model), questions: QUESTIONS }
  return JSON.stringify(local ? { ...body, keep_alive: '30m' } : body)
}

function count(rx: RegExp, text: string): number {
  rx.lastIndex = 0
  return text.match(rx)?.length ?? 0
}

export function features(prompt: string, a: Answers): Features {
  const t = a.tier.probabilities
  const text = trim(prompt, 4000)
  const levels = QUESTIONS.effort.criteria.length - 1
  const top = TIERS.length - 1
  const f: Features = {
    // The expected rank of the tier the model would pick, in [0, 1].
    tier: TIERS.reduce((s, name, i) => s + (t[name] ?? 0) * (i / top), 0),
    simple: a.simple.noul,
    deep: a.deep.noul,
    frontier: a.frontier?.noul ?? 0,
    effort: (a.effort.score ?? 0) / levels,
    long: Math.min(text.length / 1500, 1),
  }
  let harder = false
  for (const [name, rx] of Object.entries(SIGNALS)) {
    f[name] = Math.min(count(rx, text), 2) / 2
    harder ||= f[name] > 0
  }
  f.quick = QUICK.test(text) && text.length < 80 && !harder ? 1 : 0
  return f
}

const get = (r: Record<string, number>, k: string) => r[k] ?? 0

/** Difficulty in [0, 1]. */
export function score(f: Features, w: Record<string, number> = WEIGHTS): number {
  const sum = (names: readonly string[], g: (n: string) => number) =>
    names.reduce((s, n) => s + g(n), 0)
  const pos = sum(POSITIVE, n => get(w, `w_${n}`) * get(f, n))
  const neg = sum(NEGATIVE, n => get(w, `w_${n}`) * get(f, n))
  const hi = sum(POSITIVE, n => get(w, `w_${n}`))
  const lo = -sum(NEGATIVE, n => get(w, `w_${n}`))
  return hi > lo ? (pos - neg - lo) / (hi - lo) : 0.5
}

export function tierFor(s: number, w: Record<string, number> = WEIGHTS): Tier {
  for (let i = CUTS.length - 1; i >= 0; i--) if (s >= get(w, CUTS[i]!)) return TIERS[i + 1]!
  return 'haiku'
}

/** How close a score is to any threshold. */
export function edge(s: number, w: Record<string, number> = WEIGHTS): number {
  return Math.min(...CUTS.map(c => Math.abs(s - get(w, c))))
}

/** The tier to use when only `enabled` tiers may be picked: the tier itself,
 * else the nearest enabled one above it (a hard task is never sent lower while
 * a higher model is on), else the nearest below; undefined when none is on.
 * The same rule as classify.allowed. */
export function allowed(tier: Tier, enabled: ReadonlySet<Tier>): Tier | undefined {
  if (enabled.has(tier)) return tier
  const i = TIERS.indexOf(tier)
  const order = [...TIERS.slice(i + 1), ...TIERS.slice(0, i).reverse()]
  return order.find(t => enabled.has(t))
}
