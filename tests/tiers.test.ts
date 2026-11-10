// Four tiers, and only the ones turned on: a tier that is off goes to the
// nearest one above it that is on, else the nearest below.
//
//   claude plugin test .

import { expect, test } from 'claude-code/testing'

import { allowed, edge, features, score, tierFor, TIERS, type Tier } from '../hooks/classify.ts'
import { WEIGHTS } from '../hooks/model.ts'

const on = (...t: Tier[]) => new Set<Tier>(t)

test('the four tiers run from haiku to fable', async () => {
  expect(TIERS).toEqual(['haiku', 'sonnet', 'opus', 'fable'])
  const w = { ...WEIGHTS, t_sonnet: 0.3, t_opus: 0.55, t_fable: 0.8 }
  expect(tierFor(0.1, w)).toBe('haiku')
  expect(tierFor(0.4, w)).toBe('sonnet')
  expect(tierFor(0.6, w)).toBe('opus')
  expect(tierFor(0.85, w)).toBe('fable')
  expect(edge(0.79, w)).toBeLessThan(0.02)
})

test('a tier that is off goes up to the nearest one on, else down', async () => {
  expect(allowed('fable', on('haiku', 'sonnet', 'opus'))).toBe('opus')
  expect(allowed('sonnet', on('haiku', 'opus', 'fable'))).toBe('opus')
  expect(allowed('haiku', on('fable'))).toBe('fable')
  expect(allowed('opus', on('haiku', 'sonnet'))).toBe('sonnet')
  expect(allowed('opus', on('haiku', 'sonnet', 'opus', 'fable'))).toBe('opus')
  expect(allowed('opus', on())).toBe(undefined)
})

test("the model's fable answer and the frontier question raise the score", async () => {
  const base = { simple: { noul: 0 }, deep: { noul: 1 }, effort: { score: 4 } }
  const prompt = 'design the sync protocol'
  const opus = features(prompt, { ...base, tier: { probabilities: { opus: 1 } }, frontier: { noul: 0 } })
  const fable = features(prompt, { ...base, tier: { probabilities: { fable: 1 } }, frontier: { noul: 1 } })
  expect(fable.tier).toBe(1)
  expect(Math.abs(opus.tier! - 2 / 3)).toBeLessThan(1e-9)
  expect(score(fable)).toBeGreaterThan(score(opus))
})
