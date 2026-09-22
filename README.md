# tongflow-router-infistar

[TongFlow](https://github.com/tong-io/tongflow) plugin for the
[Infistar](https://infistar.cc) (无限星河AI) gateway — one key in front of GPT,
Claude, Gemini, Qwen, DeepSeek, GLM, Kimi, Doubao, MiniMax and Grok, plus image,
video and transcription models.

> **Not yet exercised against a live key.** Routes and model ids come from
> Infistar's public docs and model marketplace — see [Unverified](#unverified).

## Capabilities

Implements these ABI slots (runs locally as a Python process, no GPU):

| Slot | Route | Shortlist default |
| --- | --- | --- |
| `gen-text` / `split-text` / `combine-text` | `POST /v1/chat/completions` | `gpt-5.6-sol` |
| `arrange-group` / `drop-video` | `POST /v1/chat/completions` | `gpt-5.6-sol` |
| `image-gen-text` (image understanding) | `POST /v1/chat/completions` | `gpt-6-astra` |
| `image-gen` | `POST /v1/images/generations` | `gpt-image-2` |
| `image-edit` / `image-fusion` | `POST /v1/images/edits` | `qwen-image-edit-max` |
| `text-gen-video` / `image-gen-video` | `POST /v1/videos` → poll → `/content` | `wan3.0-video` / `wan2.7-i2v` |
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

## Unverified

Everything below was derived from Infistar's public docs and public model
marketplace, **not from a live key**. Each is a plausible failure on first run:

1. **`GET /v1/models` shape.** The live catalog assumes each record carries
   `supported_endpoint_types`, as the public marketplace does (both are
   new-api derivatives). If the key's list only returns bare ids, the dropdowns
   fall back to the shortlist — which still works, just without live extension.
2. **`/v1/images/edits` request format.** The docs say "most OpenAI-compatible
   models take `multipart/form-data`; some take image URLs as JSON". This
   implementation sends multipart, with `image[]` for multi-reference fusion.
3. **`input_reference.image_url` on `/v1/videos`.** Canvas assets are bytes, so
   they go in as a data URI. The docs describe a reference image file without
   saying whether a data URI is accepted.
4. **Non-streamed chat.** Requests are always streamed, on the assumption that
   this gateway shares the new-api family's intermittently empty non-streamed
   body for the GPT-5.x family. Harmless if it doesn't.
5. **Channel quality.** A third-party probe (veridrop) scored `infistar.ai`
   71/100 and flagged missing Claude encryption signatures. Worth a straight
   answer from the vendor about upstream channels before this is presented as
   an official integration.
