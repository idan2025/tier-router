// What the router needs from Ollama besides its answers: a release new enough
// for the chosen decision model, the model itself, an upgrade when the release
// is too old, and (for the `cpu` device) a CPU-only Ollama of its own.
//
// - Ollama has no per-request device choice for /v1/systemone: which processor
//   runs a model is the server's. So `cpu` starts a second Ollama, on
//   CPU_PORT, with every GPU hidden from it, reading the same model store;
//   `gpu` and `auto` ask the user's own Ollama, which uses a GPU when it has
//   one, and `gpu` reports when the model ended up on the CPU anyway.
// - An upgrade is never started without the user's yes, and runs only what
//   Ollama itself publishes: its install script on Linux, Homebrew on macOS
//   (when Ollama came from there), winget on Windows. When that cannot run
//   unattended (the Linux script needs root), the command to run is shown.

import type { EngineInterface as Engine } from 'claude-code'

/** Where the CPU-only Ollama listens: Ollama's own port plus one. */
export const CPU_PORT = 11435

/** Every GPU runtime's "see no devices" switch, as Ollama's hardware docs name them. */
const NO_GPU: Record<string, string> = {
  CUDA_VISIBLE_DEVICES: '-1',
  ROCR_VISIBLE_DEVICES: '-1',
  HIP_VISIBLE_DEVICES: '-1',
  GGML_VK_VISIBLE_DEVICES: '-1',
  OLLAMA_VULKAN: '0',
  OLLAMA_LLM_LIBRARY: 'cpu',
}

/** Where Linux's installer keeps the service's models (it runs as user `ollama`). */
const SERVICE_MODELS = '/usr/share/ollama/.ollama/models'

/** `0.40.5` >= `0.40.0`; a part that is not a number counts as 0. */
export function atLeast(have: string, want: string): boolean {
  const parts = (v: string) => v.replace(/^v/, '').split(/[.-]/).map(n => Number.parseInt(n, 10) || 0)
  const a = parts(have)
  const b = parts(want)
  for (let i = 0; i < Math.max(a.length, b.length); i++) {
    if ((a[i] ?? 0) !== (b[i] ?? 0)) return (a[i] ?? 0) > (b[i] ?? 0)
  }
  return true
}

async function within<T>($: Engine, ms: number, work: Promise<T>): Promise<T> {
  return Promise.race([
    work,
    $.clock.sleep(ms).then(() => {
      throw new Error(`no answer in ${ms / 1000} s`)
    }),
  ])
}

/** The Ollama release at `url`, or undefined when nothing answers there. */
export async function version($: Engine, url: string): Promise<string | undefined> {
  try {
    const res = await within($, 3000, $.http.fetch(`${url}/api/version`))
    return res.ok ? String(JSON.parse(res.text).version ?? '') || undefined : undefined
  } catch {
    return undefined
  }
}

/** Whether the Ollama at `url` has `model`. */
export async function hasModel($: Engine, url: string, model: string): Promise<boolean> {
  try {
    const res = await within($, 5000, $.http.fetch(`${url}/api/show`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ model }),
    }))
    return res.ok
  } catch {
    return false
  }
}

/** Downloads `model` into the Ollama at `url`; the error text when it could not. */
export async function pull($: Engine, url: string, model: string): Promise<string | undefined> {
  try {
    const res = await $.http.fetch(`${url}/api/pull`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ model, stream: false }),
    })
    if (res.ok) return undefined
    try {
      return String(JSON.parse(res.text).error ?? `HTTP ${res.status}`)
    } catch {
      return `HTTP ${res.status}`
    }
  } catch (err) {
    return err instanceof Error ? err.message : String(err)
  }
}

/** True when Ollama's answer says the model needs a newer Ollama. */
export function needsNewerOllama(text: string): boolean {
  return /newer version of ollama|requires? (a )?newer|upgrade ollama|update ollama/i.test(text)
}

/** Which processor holds `model` in the Ollama at `url`, once it is loaded. */
export async function processor($: Engine, url: string, model: string): Promise<'cpu' | 'gpu' | 'split' | undefined> {
  try {
    const res = await within($, 3000, $.http.fetch(`${url}/api/ps`))
    if (!res.ok) return undefined
    const want = model.includes(':') ? model : `${model}:latest`
    const m = (JSON.parse(res.text).models ?? []).find((m: { name?: string }) => m.name === want || m.name === model)
    if (!m) return undefined
    if (!m.size_vram) return 'cpu'
    return m.size_vram >= m.size ? 'gpu' : 'split'
  } catch {
    return undefined
  }
}

async function ran($: Engine, argv: string[], timeoutMs = 10000): Promise<{ code: number; out: string }> {
  try {
    const r = await $.process.run(argv, { timeoutMs })
    return { code: r.exitCode, out: `${r.stdout}\n${r.stderr}`.trim() }
  } catch (err) {
    return { code: -1, out: err instanceof Error ? err.message : String(err) }
  }
}

export type Upgrade = { argv: string[]; label: string } | { manual: string }

/** How this machine upgrades Ollama: a command the plugin may run, or what
 * the user should run themselves. */
export async function upgradePlan($: Engine): Promise<Upgrade> {
  const os = (await ran($, ['uname', '-s'])).out
  if (/linux/i.test(os)) {
    const script = 'curl -fsSL https://ollama.com/install.sh | sh'
    const root = (await ran($, ['id', '-u'])).out === '0'
    const sudo = root || (await ran($, ['sudo', '-n', 'true'])).code === 0
    return sudo ? { argv: ['sh', '-c', script], label: script } : { manual: script }
  }
  if (/darwin/i.test(os)) {
    if ((await ran($, ['brew', 'list', '--versions', 'ollama'])).code === 0) {
      return { argv: ['brew', 'upgrade', 'ollama'], label: 'brew upgrade ollama' }
    }
    return { manual: 'download the new Ollama from https://ollama.com/download and replace the app' }
  }
  if ((await ran($, ['winget', '--version'])).code === 0) {
    const argv = ['winget', 'upgrade', '--id', 'Ollama.Ollama', '-e', '--silent', '--accept-source-agreements', '--accept-package-agreements']
    return { argv, label: argv.join(' ') }
  }
  return { manual: 'download the new Ollama from https://ollama.com/download and run its installer' }
}

/** Runs an upgrade; the output's last lines when it failed. */
export async function upgrade($: Engine, plan: { argv: string[] }): Promise<string | undefined> {
  const r = await ran($, plan.argv, 600000)
  return r.code === 0 ? undefined : r.out.split('\n').slice(-5).join('\n') || `exit ${r.code}`
}

/** Starts the CPU-only Ollama unless one already answers on CPU_PORT (another
 * session's); resolves its URL, or why it could not start. It lives as long as
 * the session that started it. */
export async function cpuServer($: Engine): Promise<{ url: string } | { error: string }> {
  const url = `http://127.0.0.1:${CPU_PORT}`
  if (await version($, url)) return { url }
  const env: Record<string, string> = { ...NO_GPU, OLLAMA_HOST: `127.0.0.1:${CPU_PORT}`, OLLAMA_NOPRUNE: '1' }
  // A Linux service install keeps its models in its own home; read them there
  // rather than download a second copy into the user's.
  if (await $.fs.exists(`${SERVICE_MODELS}/manifests`).catch(() => false)) env.OLLAMA_MODELS = SERVICE_MODELS
  let failed: string | undefined
  void (async () => {
    try {
      const child = $.process.spawn({ argv: ['ollama', 'serve'], env })
      for await (const { text } of child) $.ui.log(text.trimEnd(), { to: 'debug' })
    } catch (err) {
      failed = err instanceof Error ? err.message : String(err)
    }
  })()
  for (let i = 0; i < 40 && !failed; i++) {
    await $.clock.sleep(250)
    if (await version($, url)) return { url }
  }
  return { error: failed ? `could not start ollama (${failed})` : `the CPU Ollama on port ${CPU_PORT} did not answer` }
}

export type Local = { name: string; digest: string; cloud: boolean }

/** The models the Ollama at `url` holds, from /api/tags. A cloud model (its
 * `remote_model` set) runs on Ollama's servers and has no local build to update. */
export async function localModels($: Engine, url: string): Promise<Local[]> {
  try {
    const res = await within($, 5000, $.http.fetch(`${url}/api/tags`))
    if (!res.ok) return []
    const models = JSON.parse(res.text).models ?? []
    return models.map((m: { name?: string; digest?: string; remote_model?: string }) => ({
      name: String(m.name ?? ''),
      digest: String(m.digest ?? '').replace(/^sha256:/, ''),
      cloud: Boolean(m.remote_model),
    }))
  } catch {
    return []
  }
}

/** The digest the public registry has for `name`'s tag, or undefined when it
 * does not answer: offline, a private host, or a name it does not serve. */
export async function registryDigest($: Engine, name: string): Promise<string | undefined> {
  const [base = '', tag = 'latest'] = name.split(':')
  const parts = base.split('/')
  if (parts.length > 2 || (parts.length === 2 && parts[0]?.includes('.'))) return undefined
  const ns = parts.length === 2 ? parts[0] : 'library'
  const repo = parts[parts.length - 1]
  try {
    const res = await within($, 8000, $.http.fetch(`https://registry.ollama.com/v2/${ns}/${repo}/manifests/${tag}`, {
      method: 'HEAD',
      headers: { accept: 'application/vnd.docker.distribution.manifest.v2+json' },
    }))
    const digest = res.ok ? res.headers['ollama-content-digest'] : undefined
    return digest ? digest.replace(/^sha256:/, '') : undefined
  } catch {
    return undefined
  }
}

/** Local models whose published build is not the one held: name and the new digest. */
export async function staleModels($: Engine, url: string): Promise<{ name: string; to: string }[]> {
  const stale: { name: string; to: string }[] = []
  for (const m of await localModels($, url)) {
    if (m.cloud || !m.digest) continue
    const to = await registryDigest($, m.name)
    if (to && to !== m.digest) stale.push({ name: m.name, to })
  }
  return stale
}
