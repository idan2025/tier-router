// The plugin's setup against a fake Ollama: an upgrade only after a yes, a
// download only after a yes, a declined offer not made again at the next
// session start, and the `cpu` device asking a CPU-only Ollama of its own.
//
//   claude plugin test .

import type { On } from 'claude-code'
import { expect, mock, test } from 'claude-code/testing'

type Fake = { version: string; models: string[]; asked: string[]; pulled: string[]; ran: string[][]; spawned: Record<string, string>[]; urls: string[]; local: Record<string, string>; registry: Record<string, string>; pullError?: string }

/** A fake Ollama on both ports, the host's commands, the user's answers. */
function world(on: On, yes: boolean, fake: Fake, sudo = true) {
  mock.store(on)
  on('command.register', () => ({ value: { command: 'router' } }))
  on('session.start', (_, e) => ({ cwd: e.cwd }))
  on('ui.status', () => ({ value: undefined }))
  on('ui.toast', () => ({ value: undefined }))
  on('ui.log', () => ({ value: undefined }))
  on('fs.exists', () => ({ value: false }))
  // Short waits pass at once; the long ones are timeouts that never fire here.
  on('clock.sleep', (_, e) => (e.ms < 2000 ? { value: undefined } : new Promise(() => {})))
  on('clock.now', () => ({ value: 0 }))
  on('tool.call', (_, e) => {
    if (e.tool !== 'AskUserQuestion') return { deny: 'not here' }
    const q = e.questions[0]!
    fake.asked.push(q.question)
    return { result: { questions: e.questions, answers: { [q.question]: yes ? q.options[0]!.label : 'Not now' } } }
  })
  on('process.run', (_, e) => {
    fake.ran.push([...e.argv])
    const out = (stdout: string, exitCode = 0) => ({ value: { exitCode, stdout, stderr: '' } })
    if (e.argv[0] === 'uname') return out('Linux')
    if (e.argv[0] === 'id') return out('1000')
    if (e.argv[0] === 'sudo') return out('', sudo ? 0 : 1)
    if (e.argv[0] === 'sh') {
      fake.version = '0.40.5'
      return out('installed')
    }
    return out('', 1)
  })
  on('process.spawn', async function* (_, e) {
    fake.spawned.push({ ...e.env })
    return { value: { code: 0, signal: null } }
  })
  on('http.fetch', (_, e) => {
    fake.urls.push(e.url)
    const ok = (body: unknown) => ({ value: { status: 200, ok: true, headers: {}, text: JSON.stringify(body) } })
    const cpuUp = fake.spawned.length > 0
    if (e.url.startsWith('http://127.0.0.1:11435') && !cpuUp) return { deny: 'connection refused' }
    const path = new URL(e.url).pathname
    if (path === '/api/version') return ok({ version: fake.version })
    if (path === '/api/show') {
      const { model } = JSON.parse(e.init?.body ?? '{}')
      return fake.models.includes(model) ? ok({}) : { value: { status: 404, ok: false, headers: {}, text: '{"error":"not found"}' } }
    }
    if (path === '/api/pull') {
      const { model } = JSON.parse(e.init?.body ?? '{}')
      if (fake.pullError) return { value: { status: 500, ok: false, headers: {}, text: JSON.stringify({ error: fake.pullError }) } }
      fake.pulled.push(model)
      fake.models.push(model)
      return ok({ status: 'success' })
    }
    if (path === '/api/tags') return ok({ models: Object.entries(fake.local).map(([name, digest]) => ({ name, digest, ...(name.endsWith(':cloud') ? { remote_model: name } : {}) })) })
    if (e.url.startsWith('https://registry.ollama.com/')) {
      // /v2/<namespace>/<repo>/manifests/<tag>
      const seg = new URL(e.url).pathname.split('/')
      const digest = fake.registry[`${seg[seg.length - 3]}:${seg[seg.length - 1]}`]
      return digest ? { value: { status: 200, ok: true, headers: { 'ollama-content-digest': digest }, text: '' } } : { value: { status: 404, ok: false, headers: {}, text: '' } }
    }
    if (path === '/api/ps') return ok({ models: [{ name: 'laya:latest', size: 100, size_vram: 0 }] })
    if (path === '/v1/systemone') return ok({ answers: {} })
    return { value: { status: 404, ok: false, headers: {}, text: '' } }
  })
}

// Every local model starts as the registry's current build (digest all a's).
const fresh = (version: string, models: string[]): Fake => {
  const local = Object.fromEntries(models.map(m => [m.includes(':') ? m : `${m}:latest`, 'a'.repeat(64)]))
  const registry = Object.fromEntries(Object.keys(local).map(k => [k, 'a'.repeat(64)]))
  return { version, models, asked: [], pulled: [], ran: [], spawned: [], urls: [], local, registry }
}

const start = { cwd: '/', surface: 'terminal', isInteractive: true } as const
const setup = { command: 'router', args: 'setup' } as const

test('an Ollama too old for laya is upgraded only after a yes, then laya is downloaded', { options: { tev1Model: 'laya' } }, async ($, on) => {
  const fake = fresh('0.35.2', [])
  world(on, true, fake)
  const done = await $.command.run(setup)
  expect(fake.asked[0]).toContain('needs Ollama 0.40.0 or newer')
  expect(fake.ran).toContainEqual(['sh', '-c', 'curl -fsSL https://ollama.com/install.sh | sh'])
  expect(fake.asked[1]).toContain('Download it now')
  expect(fake.pulled).toEqual(['laya'])
  expect(done.text).toContain('laya is ready')
})

test('without root or sudo, the upgrade is shown rather than run', { options: { tev1Model: 'laya' } }, async ($, on) => {
  const fake = fresh('0.35.2', ['laya'])
  world(on, true, fake, false)
  const done = await $.command.run(setup)
  expect(fake.asked).toEqual([])
  expect(done.text).toContain('To upgrade, run: curl -fsSL https://ollama.com/install.sh | sh')
})

test('a declined download is not offered again at the next session start', { options: { tev1Model: 'laya' } }, async ($, on) => {
  const fake = fresh('0.40.5', [])
  world(on, false, fake)
  const first = await $.command.run(setup)
  expect(first.text).toContain('Ollama has no laya')
  expect(fake.asked.length).toBe(1)
  await $.session.start(start)
  await new Promise(resolve => setTimeout(resolve, 100))
  expect(fake.asked.length).toBe(1)
  expect(fake.pulled).toEqual([])
})

test('the cpu device asks a CPU-only Ollama with every GPU hidden', { options: { tev1Model: 'laya', device: 'cpu' } }, async ($, on) => {
  const fake = fresh('0.40.5', ['laya'])
  world(on, false, fake)
  const done = await $.command.run(setup)
  expect(fake.spawned.length).toBe(1)
  expect(fake.spawned[0]).toMatchObject({ OLLAMA_HOST: '127.0.0.1:11435', CUDA_VISIBLE_DEVICES: '-1', OLLAMA_VULKAN: '0' })
  expect(fake.urls).toContain('http://127.0.0.1:11435/v1/systemone')
  // The model is never loaded into the GPU Ollama it was kept off.
  expect(fake.urls).not.toContain('http://localhost:11434/v1/systemone')
  expect(done.text).toContain('in the CPU-only Ollama at http://127.0.0.1:11435')
})

test('the gpu device says when the model ended up on the CPU', { options: { tev1Model: 'laya', device: 'gpu' } }, async ($, on) => {
  const fake = fresh('0.40.5', ['laya'])
  world(on, false, fake)
  const done = await $.command.run(setup)
  expect(fake.spawned).toEqual([])
  expect(done.text).toContain('Ollama found no GPU')
})

test('a build from source (0.0.0) is never offered an upgrade', { options: { tev1Model: 'laya' } }, async ($, on) => {
  const fake = fresh('0.0.0', ['laya'])
  world(on, true, fake)
  const done = await $.command.run(setup)
  expect(fake.asked).toEqual([])
  expect(fake.ran.filter(argv => argv[0] === 'sh')).toEqual([])
  expect(done.text).toContain('laya is ready')
})

test('an Ollama on another machine is never upgraded from here', { options: { tev1Model: 'laya', ollamaUrl: 'http://gpu-box:11434' } }, async ($, on) => {
  const fake = fresh('0.35.2', ['laya'])
  world(on, true, fake)
  const done = await $.command.run(setup)
  expect(fake.asked).toEqual([])
  expect(fake.ran).toEqual([])
  expect(done.text).toContain('Upgrade it there')
})

test('a local model with a newer registry build is offered an update, and it is pulled on yes', { options: { tev1Model: 'laya' } }, async ($, on) => {
  const fake = fresh('0.40.5', ['laya'])
  fake.registry['laya:latest'] = 'b'.repeat(64)
  world(on, true, fake)
  const done = await $.command.run(setup)
  expect(fake.asked.some(q => q.includes('newer builds exist for laya:latest'))).toBe(true)
  expect(fake.pulled).toEqual(['laya:latest'])
  expect(done.text).toContain('Updated laya:latest.')
})

test('a model already on the newest build is not offered an update', { options: { tev1Model: 'laya' } }, async ($, on) => {
  const fake = fresh('0.40.5', ['laya'])
  world(on, true, fake)
  await $.command.run(setup)
  expect(fake.asked.some(q => q.includes('newer builds'))).toBe(false)
  expect(fake.pulled).toEqual([])
})

test('a cloud model is never checked or offered an update', { options: { tev1Model: 'laya' } }, async ($, on) => {
  const fake = fresh('0.40.5', ['laya'])
  fake.local['glm-5.3:cloud'] = 'c'.repeat(64)
  fake.registry['glm-5.3:cloud'] = 'd'.repeat(64)
  world(on, true, fake)
  await $.command.run(setup)
  expect(fake.asked.some(q => q.includes('glm-5.3'))).toBe(false)
  expect(fake.urls.some(u => u.includes('registry.ollama.com') && u.includes('glm'))).toBe(false)
})

test('offline, no update is offered and setup still finishes', { options: { tev1Model: 'laya' } }, async ($, on) => {
  const fake = fresh('0.40.5', ['laya'])
  fake.registry = {}
  world(on, true, fake)
  const done = await $.command.run(setup)
  expect(fake.asked.some(q => q.includes('newer builds'))).toBe(false)
  expect(done.text).toContain('laya is ready')
})

test('a declined update is not offered again until the build changes', { options: { tev1Model: 'laya' } }, async ($, on) => {
  const fake = fresh('0.40.5', ['laya'])
  fake.registry['laya:latest'] = 'b'.repeat(64)
  world(on, false, fake)
  await $.command.run(setup)
  await $.session.start(start)
  await new Promise(resolve => setTimeout(resolve, 100))
  expect(fake.asked.filter(q => q.includes('newer builds')).length).toBe(1)
  expect(fake.pulled).toEqual([])
})

test('tev1 gets the same update offer as laya', { options: { tev1Model: 'tev1' } }, async ($, on) => {
  const fake = fresh('0.40.5', ['tev1'])
  fake.registry['tev1:latest'] = 'e'.repeat(64)
  world(on, true, fake)
  const done = await $.command.run(setup)
  expect(fake.asked.some(q => q.includes('newer builds exist for tev1:latest'))).toBe(true)
  expect(fake.pulled).toEqual(['tev1:latest'])
  expect(done.text).toContain('Updated tev1:latest.')
})

test('a failed update leaves the decision model ready and names the model', { options: { tev1Model: 'laya' } }, async ($, on) => {
  const fake = fresh('0.40.5', ['laya', 'other'])
  fake.registry['other:latest'] = 'f'.repeat(64)
  fake.pullError = 'disk full'
  world(on, true, fake)
  const done = await $.command.run(setup)
  expect(done.text).toContain('laya is ready')
  expect(done.text).toContain('Updating failed for other:latest (disk full)')
})

test('two declined offers are both remembered', { options: { tev1Model: 'laya' } }, async ($, on) => {
  const fake = fresh('0.40.5', ['laya', 'other'])
  fake.registry['other:latest'] = 'f'.repeat(64)
  world(on, false, fake)
  await $.command.run(setup)
  fake.registry['laya:latest'] = 'b'.repeat(64)
  await $.command.run(setup)
  fake.registry = { 'laya:latest': 'a'.repeat(64), 'other:latest': 'f'.repeat(64) }
  await $.session.start(start)
  await new Promise(resolve => setTimeout(resolve, 100))
  // other's build was declined first, then a second offer: neither comes back at session start.
  expect(fake.asked.filter(q => q.includes('newer builds')).length).toBe(2)
})
