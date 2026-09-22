# tongflow-router-infistar

[TongFlow](https://github.com/tong-io/tongflow) plugin for the
[Infistar](https://infistar.cc) (无限星河AI) gateway — one key in front of GPT,
Claude, Gemini, Qwen, DeepSeek, GLM, Kimi, Doubao, MiniMax and Grok, plus image,
video and transcription models.

> Exercised end to end against a live key. See
> [Request shapes](#request-shapes) for how this plugin talks to the gateway.

## Capabilities

Implements these ABI slots (runs locally as a Python process, no GPU):

| Slot | Route | Default |
| --- | --- | --- |
| `gen-text` / `split-text` / `combine-text` | `POST /v1/chat/completions` | `gpt-5.6-sol` |
| `arrange-group` / `drop-video` | local, no LLM call | — |
| `image-gen-text` (image understanding) | `POST /v1/chat/completions` | `gpt-6-astra` |
| `image-gen` | `POST /v1/images/generations` | `gpt-image-2` |
| `image-edit` | `POST /v1/images/edits` | `step-image-edit-2` |
| `image-fusion` | `POST /v1/images/edits` (`image[]`) | `gpt-image-2` |
| `text-gen-video` | `POST /v1/videos` → poll → `/content` | `wan3.0-video` |
| `image-gen-video` | `POST /v1/videos` → poll → `/content` | `MiniMax-H3` |
| `transcribe` | `POST /v1/audio/transcriptions` | `qwen-audio-3.0-asr-flash-filetrans` |

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

## Request shapes

How this plugin calls the gateway, in case you are extending it:

- **Video** (`/v1/videos`) is submitted without `seconds` / `size` — the model's
  own defaults apply. `INFISTAR_VIDEO_SECONDS` / `INFISTAR_VIDEO_SIZE` send them
  explicitly when a model accepts a spec.
- **`input_reference`** is a plain string: a URL or a data URI. Canvas assets go
  in as a data URI, since the gateway has no upload endpoint.
- **Image edits** (`/v1/images/edits`) are multipart, without `size`; an edit
  keeps the source image's geometry. A multi-reference edit repeats `image[]`
  per source and needs a model with a multi-image channel, which is why
  `image-fusion` defaults to `gpt-image-2` while single-image edits default to
  the faster `step-image-edit-2`.
- **Chat** is always streamed; the non-streaming path can come back empty for
  some model families.
- **Model ids** come from `GET /v1/models`, which carries
  `supported_endpoint_types` per record — that is what drives the per-slot
  dropdowns, so the list always matches what your key's group allows.
- **`transcribe`** needs a key whose group includes an `audio-transcription`
  model.
