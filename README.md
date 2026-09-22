# tongflow-router-infistar

[TongFlow](https://github.com/tong-io/tongflow) plugin for the
[Infistar](https://infistar.cc) (无限星河AI) gateway — one key in front of GPT,
Claude, Gemini, Qwen, DeepSeek, GLM, Kimi, Doubao, MiniMax and Grok, plus image,
video and transcription models.

> **Exercised against a live key on 2026-09-22.** Ten of twelve slots produce
> real output; see [Verified](#verified) for what the gateway actually accepts
> and the two slots that need something from Infistar.

## Capabilities

Implements these ABI slots (runs locally as a Python process, no GPU):

| Slot | Route | Default | Measured |
| --- | --- | --- | --- |
| `gen-text` / `split-text` / `combine-text` | `POST /v1/chat/completions` | `gpt-5.6-sol` | 3–17s |
| `arrange-group` / `drop-video` | local, no LLM call | — | instant |
| `image-gen-text` (image understanding) | `POST /v1/chat/completions` | `gpt-6-astra` | 16s |
| `image-gen` | `POST /v1/images/generations` | `gpt-image-2` | 246s |
| `image-edit` | `POST /v1/images/edits` | `step-image-edit-2` | 16s |
| `image-fusion` | `POST /v1/images/edits` (`image[]`) | `gpt-image-2` | 643s |
| `text-gen-video` | `POST /v1/videos` → poll → `/content` | `wan3.0-video` | 203s |
| `image-gen-video` | `POST /v1/videos` → poll → `/content` | `MiniMax-H3` | 219s |
| `transcribe` | `POST /v1/audio/transcriptions` | `qwen-audio-3.0-asr-flash-filetrans` | needs an ASR-enabled key |

**Text-to-speech is not implemented.** The gateway answers on `/v1/audio/speech`,
but its catalog lists no TTS model, so there would be nothing to select.

## Models

Each node has a **model dropdown**. It starts from the curated shortlist above
and is extended live from `GET /v1/models` — the key's own list, so it reflects
what your token's group actually allows. Any id that list knows is accepted at
run time, so a model added after this plugin shipped needs no update here.

The catalog has 198 models at the time of writing; the public marketplace at
[infistar.cc/pricing](https://infistar.cc/pricing) is the full picture.

## Node knobs

| Control | Where | Notes |
| --- | --- | --- |
| Temperature / Reasoning effort | **Advanced**, text + vision nodes | Reasoning effort applies to reasoning models only |
| Quality | **Advanced**, image nodes | Only models that document it |
| Seconds | **Advanced**, video nodes | The node's own `duration` wins when set |

Width / height snap to the closest size the route accepts.

## Credentials

Add in TongFlow **Settings** (gear icon, top-right):

| Key | Required | Notes |
| --- | --- | --- |
| `INFISTAR_API_KEY` | ✅ | Create one under 令牌管理 at [infistar.cc](https://infistar.cc). |
| `INFISTAR_BASE_URL` | optional | Defaults to `https://infistar.cc/v1`; `https://infistar.ai/v1` serves the same account. |
| `INFISTAR_POLL_TIMEOUT_S` | optional | Max seconds to wait for a video task (default `900`). |

## Verified

Run against a live key on 2026-09-22. What the gateway turned out to want,
none of which is in its docs:

1. **The video route rejects every `seconds` / `size` combination** with
   「当前模型没有支持本次媒体规格的可用渠道」 — including the values from its
   own doc examples. The identical request *without* them is accepted, so this
   plugin sends neither and lets the model pick. `INFISTAR_VIDEO_SECONDS` /
   `INFISTAR_VIDEO_SIZE` force them back for the day a channel accepts a spec.
2. **`input_reference` is a plain string**, not OpenAI's `{"image_url": …}`
   object — the object form is rejected by the gateway's Go decoder. A data URI
   works; there is no upload endpoint (`/v1/files` exists but wants a model).
3. **The image-edit route also rejects `size`**, the same way.
4. **Multi-reference edits need `image[]`** (repeating plain `image` fails with
   "image file is required") **and a model with a multi-image channel** —
   `step-image-edit-2` routes single-image edits only, so `image-fusion`
   defaults to `gpt-image-2`.
5. **`wan2.7-i2v` fails every image-to-video task**, with a data URI or a public
   URL alike (`status: failed`, quota refunded), which is why `MiniMax-H3` is
   the `image-gen-video` default.
6. **`GET /v1/models` does carry `supported_endpoint_types`** on every record,
   so the live dropdown extension works as designed.
7. Failed tasks refund their frozen quota (`billing_status: REFUNDED`).

### Needs something from Infistar

- **`transcribe`**: the test key's group exposes **no** `audio-transcription`
  model at all, though the public marketplace lists three. The slot is wired
  and will work on a key whose group includes them.
- **`drop-video`** returns an empty `clips` list. That is an upstream SDK bug,
  not this plugin: `drop_video_output()` reads `prompt["fileKeys"]` while the
  slot's ABI input field is `videos`, and `@node_slot` forbids extras. Every
  plugin using that helper is affected.

### Slow paths

`gpt-image-2` is the slowest route here by a wide margin — 246s for a single
generation, 643s for a two-image fusion. `step-image-edit-2` does a single edit
in 16s. Pick accordingly.
