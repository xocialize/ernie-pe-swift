# ernie-pe-swift

The **ERNIE Prompt Enhancer** (Ministral-3B-Instruct fine-tune, Apache-2.0, ships in
[baidu/ERNIE-Image-Turbo](https://huggingface.co/baidu/ERNIE-Image-Turbo)'s `pe/` dir) as a
**standalone MLXEngine `llm` package** — the engine's first `llm` surface.

Deliberately independent of every text-to-image package: prompt enhancement is an LLM **mode**
(C4), not a tool. Its text output drives **any** registered `textToImage` backer (Lens,
ERNIE-Turbo, future models) — or no t2i at all (it is a competent general 3B chat model).

- **`ErniePE`** — the standalone port. Backbone = the same Mistral3 architecture parity-locked in
  [ernie-image-swift](https://github.com/xocialize/ernie-image-swift) (26L / 3072 / GQA 32-8 /
  head 128, YaRN rope factor 16), **vendored here** (not a package dependency, so this repo stands
  alone) + KV cache + top-p sampling + lm_head.
- **`MLXErniePE`** — the MLXEngine wrapper (`ErniePEPackage`, PackageID `ernie-pe-3b`): the
  canonical `LLMRequest`/`LLMResponse` surface with modes `[promptEnhance, direct]`.

## Modes

| Mode | Behavior |
|---|---|
| `promptEnhance` | Last user turn = a **brief** image prompt; `metaData` `width`/`height` carry the target resolution → response is a rich, t2i-ready visual description for any `textToImage` model. |
| `direct` (default) | General chat over the canonical messages (system + alternating user/assistant turns). |

**Template ownership (the make-or-break):** the checkpoint's own chat template hardcodes a Chinese
system prompt and drops system messages — English in → Chinese out, and in-prompt hints do not
steer. The package owns the template (standard Mistral control tokens; a validated English enhance
instruction; a `system` `ChatMessage` overrides it, e.g. for Chinese-prompt models). English
following is sampling-brittle, so the enhance path also **seeds the assistant turn** with a short
prefix (`"A "` by default; override via `metaData["responsePrefix"]`, or disable by supplying a
custom system instruction).

## Use

```swift
import MLXErniePE
import MLXToolKit

let package = ErniePEPackage(configuration: .init(
    quantizedPath: "<root>/ERNIE-PE-3B-4bit"))   // nil → bf16 from weightsPath/tokenizerPath
try await package.load()
let response = try await package.run(LLMRequest(
    prompt: "a lighthouse on a stormy coast at dusk",
    mode: .promptEnhance,
    metaData: ["width": .int(1024), "height": .int(1024)])) as! LLMResponse
// response.text → feed straight into a T2IRequest on any textToImage backer
```

int4: 2.5 GB disk / **2.7 GB resident**, ~5.2 s load+enhance. bf16 ≈ 9 GB resident. PE +
ERNIE-Turbo-4bit ≈ 10 GB — an `llm` enhancer and a `textToImage` model **co-resident** on one
engine (separate capability slots).

## Status / consuming this package

Weights are **not yet on the Hub** — load from a local `pe/` + `pe_tokenizer/` snapshot (or a
converted 4-bit export). The package depends on **`mlx-engine-swift`** (the `MLXToolKit` contract)
via a local sibling path (`.package(path: "../mlx-engine-swift")`); clone it as a sibling to build.
No t2i dependency.

Apache-2.0 (weights) · MIT (port code).
