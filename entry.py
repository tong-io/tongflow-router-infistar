"""TongFlow plugin for Infistar (无限星河AI, https://infistar.cc) — one key in
front of GPT, Claude, Gemini, Qwen, DeepSeek, GLM, Kimi, Doubao, MiniMax, Wan,
Seedream and Seedance. Routes used (all OpenAI-compatible):

- ``POST /v1/chat/completions``        text and vision
- ``POST /v1/images/generations``      text → image
- ``POST /v1/images/edits``            image(s) + text → image
- ``POST /v1/videos``                  text / reference image → video (async task)
- ``GET  /v1/videos/{id}``             poll a video task
- ``GET  /v1/videos/{id}/content``     download the finished mp4
- ``POST /v1/audio/transcriptions``    audio → text

Model choice is per node: ``TONGFLOW_SLOT_MODELS`` is the curated shortlist
(first entry = default) and ``TONGFLOW_MODEL_CATALOG`` extends each dropdown
from the key's own model list (``GET /v1/models``, proxied by the app because
it needs the bearer token). At run time any id that list knows is accepted.

Not implemented: text-to-speech. The gateway answers on ``/v1/audio/speech``
but its catalog lists no TTS model, so there is nothing to select.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from tongflow.llm_batch_handlers import arrange_group_output, drop_video_output
from tongflow.node_slots import NodeSlots
from tongflow.progress import progress
from tongflow.protocol import Asset, asset
from tongflow.slots import current_params, node_slot
from tongflow.models.arrange_group import ArrangeGroupInput, ArrangeGroupOutput
from tongflow.models.combine_text import CombineTextInput, CombineTextOutput
from tongflow.models.drop_video import DropVideoInput, DropVideoOutput
from tongflow.models.gen_text import GenTextInput, GenTextOutput
from tongflow.models.image_edit import ImageEditInput, ImageEditOutput
from tongflow.models.image_fusion import ImageFusionInput, ImageFusionOutput
from tongflow.models.image_gen import ImageGenInput, ImageGenOutput
from tongflow.models.image_gen_text import ImageGenTextInput, ImageGenTextOutput
from tongflow.models.image_gen_video import ImageGenVideoInput, ImageGenVideoOutput
from tongflow.models.split_text import SplitTextInput, SplitTextOutput
from tongflow.models.text_gen_video import TextGenVideoInput, TextGenVideoOutput
from tongflow.models.transcribe import TranscribeInput, TranscribeOutput

# ── Per-node model picker ───────────────────────────────────────────────────
# Pure dict literal read by the platform scanner via AST (no variable
# references — each list is repeated verbatim on purpose). First entry per slot
# = default. This is only the curated shortlist; TONGFLOW_MODEL_CATALOG below
# extends each dropdown from the key's live model list, and _active_model()
# accepts any id that list exposes.
TONGFLOW_SLOT_MODELS = {
    "gen-text": ["gpt-5.6-sol", "gpt-6-astra", "claude-opus-5", "claude-sonnet-5", "gemini-3.8-flash", "deepseek-v4-flash", "qwen3.8-max", "glm-5.3", "kimi-k3", "MiniMax-M3", "grok-4.6"],
    "split-text": ["gpt-5.6-sol", "gpt-6-astra", "claude-opus-5", "claude-sonnet-5", "gemini-3.8-flash", "deepseek-v4-flash", "qwen3.8-max", "glm-5.3", "kimi-k3", "MiniMax-M3", "grok-4.6"],
    "combine-text": ["gpt-5.6-sol", "gpt-6-astra", "claude-opus-5", "claude-sonnet-5", "gemini-3.8-flash", "deepseek-v4-flash", "qwen3.8-max", "glm-5.3", "kimi-k3", "MiniMax-M3", "grok-4.6"],
    "arrange-group": ["gpt-5.6-sol", "claude-sonnet-5", "gemini-3.8-flash", "deepseek-v4-flash", "qwen3.8-max"],
    "drop-video": ["gpt-5.6-sol", "claude-sonnet-5", "gemini-3.8-flash", "deepseek-v4-flash", "qwen3.8-max"],
    "image-gen-text": ["gpt-6-astra", "gpt-5.5", "claude-opus-5", "claude-sonnet-5", "gemini-3.1-pro-preview", "qwen3.7-plus"],
    "image-gen": ["gpt-image-2", "qwen-image-2.0-pro", "wan2.7-image-pro", "doubao-seedream-5-0-260128", "gpt-image-2.5-flare", "step-2x-large"],
    "image-edit": ["step-image-edit-2", "gpt-image-2", "qwen-image-edit-max", "qwen-image-edit-plus"],
    "image-fusion": ["gpt-image-2", "gpt-image-2.5-flare", "qwen-image-edit-max"],
    "text-gen-video": ["wan3.0-video", "wan2.7-t2v", "doubao-seedance-2-5-260628", "MiniMax-H3", "doubao-seedance-2.0-mini"],
    "image-gen-video": ["MiniMax-H3", "doubao-seedance-2-5-260628", "wan3.0-video", "wan2.7-i2v"],
    "transcribe": ["qwen-audio-3.0-asr-flash-filetrans", "stepaudio-2.5-asr"],
}

# Live catalog for the canvas dropdowns. The gateway is a new-api derivative,
# so `/v1/models` carries `supported_endpoint_types` per record, same as the
# public model marketplace. It needs the bearer token, hence authEnv. A record
# matches a slot when every token is a substring of the named field (`!` negates).
TONGFLOW_MODEL_CATALOG = {
    "url": "https://infistar.cc/v1/models",
    "authEnv": "INFISTAR_API_KEY",
    "items": "data",
    "id": "id",
    "slots": {
        "gen-text": {"supported_endpoint_types": ["openai", "!image-generation", "!/v1/videos", "!audio-transcription", "!embeddings"]},
        "split-text": {"supported_endpoint_types": ["openai", "!image-generation", "!/v1/videos", "!audio-transcription", "!embeddings"]},
        "combine-text": {"supported_endpoint_types": ["openai", "!image-generation", "!/v1/videos", "!audio-transcription", "!embeddings"]},
        "arrange-group": {"supported_endpoint_types": ["openai", "!image-generation", "!/v1/videos", "!audio-transcription", "!embeddings"]},
        "drop-video": {"supported_endpoint_types": ["openai", "!image-generation", "!/v1/videos", "!audio-transcription", "!embeddings"]},
        "image-gen-text": {"supported_endpoint_types": ["openai", "!image-generation", "!/v1/videos"], "tags": "图像理解"},
        "image-gen": {"supported_endpoint_types": "image-generation"},
        "image-edit": {"supported_endpoint_types": "image-generation"},
        "image-fusion": {"supported_endpoint_types": "image-generation"},
        "text-gen-video": {"supported_endpoint_types": "/v1/videos"},
        "image-gen-video": {"supported_endpoint_types": "/v1/videos"},
        "transcribe": {"supported_endpoint_types": "audio-transcription"},
    },
}

# Per-run knobs under the node's collapsed "Advanced" section. Pure literal
# (AST-scanned); values reach the handlers through current_params().
TONGFLOW_SLOT_PARAMS = {
    "gen-text": {
        "temperature": {"type": "number", "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1, "label": "Temperature"},
        "reasoning_effort": {"type": "select", "options": ["default", "minimal", "low", "medium", "high"], "default": "default", "label": "Reasoning effort", "description": "Reasoning models only; default = the model's own."},
    },
    "split-text": {
        "temperature": {"type": "number", "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1, "label": "Temperature"},
        "reasoning_effort": {"type": "select", "options": ["default", "minimal", "low", "medium", "high"], "default": "default", "label": "Reasoning effort", "description": "Reasoning models only; default = the model's own."},
    },
    "combine-text": {
        "temperature": {"type": "number", "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1, "label": "Temperature"},
        "reasoning_effort": {"type": "select", "options": ["default", "minimal", "low", "medium", "high"], "default": "default", "label": "Reasoning effort", "description": "Reasoning models only; default = the model's own."},
    },
    "image-gen-text": {
        "temperature": {"type": "number", "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1, "label": "Temperature"},
        "reasoning_effort": {"type": "select", "options": ["default", "minimal", "low", "medium", "high"], "default": "default", "label": "Reasoning effort", "description": "Reasoning models only; default = the model's own."},
    },
    "image-gen": {
        "quality": {"type": "select", "options": ["auto", "low", "medium", "high"], "default": "auto", "label": "Quality", "description": "Only models that document it."},
    },
    "image-edit": {
        "quality": {"type": "select", "options": ["auto", "low", "medium", "high"], "default": "auto", "label": "Quality", "description": "Only models that document it."},
    },
}

# Plugin logs go to stderr — stdout is reserved for the ABI JSON response.
logging.basicConfig(
    level=os.environ.get("TONGFLOW_PLUGIN_LOG_LEVEL", "INFO").upper(),
    stream=sys.stderr,
    format="[infistar] %(levelname)s %(message)s",
)
log = logging.getLogger("tongflow.plugins.infistar")

# infistar.cc is the China-facing origin; infistar.ai serves the same account.
DEFAULT_BASE_URL = "https://infistar.cc/v1"
DEFAULT_POLL_TIMEOUT_S = 900.0
POLL_INITIAL_DELAY_S = 5.0
POLL_INTERVAL_S = 8.0
DEFAULT_IMAGE_SIZE = "1024x1024"
DEFAULT_VIDEO_SIZE = "1280x720"
DEFAULT_VIDEO_SECONDS = 5
# Sizes the video route accepts, per the gateway's docs examples.
VIDEO_SIZES = ["1280x720", "720x1280", "1024x1024", "1792x1024", "1024x1792"]
IMAGE_SIZES = ["1024x1024", "1536x1024", "1024x1536", "1792x1024", "1024x1792"]
CHAT_EMPTY_RETRIES = 2

# Model chosen on the node; set by main() from the request envelope.
_REQUEST_MODEL: str = ""


def _adv(name: str, default):
    """Advanced-section override (``TONGFLOW_SLOT_PARAMS``) or the plugin default."""
    v = current_params().get(name)
    if v is None:
        return default
    if isinstance(default, bool):
        return bool(v)
    if isinstance(default, int):
        return int(v)
    if isinstance(default, float):
        return float(v)
    return v


def _env(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def _require_api_key() -> str:
    api_key = _env("INFISTAR_API_KEY")
    if not api_key:
        raise RuntimeError(
            "INFISTAR_API_KEY is not set. Create one under 令牌管理 at "
            "https://infistar.cc and add it in TongFlow Settings."
        )
    return api_key


def _base_url() -> str:
    return (_env("INFISTAR_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


def _poll_timeout() -> float:
    raw = _env("INFISTAR_POLL_TIMEOUT_S")
    try:
        return float(raw) if raw else DEFAULT_POLL_TIMEOUT_S
    except ValueError:
        return DEFAULT_POLL_TIMEOUT_S


# ── HTTP ───────────────────────────────────────────────────────────────────


class _HttpStatus(RuntimeError):
    def __init__(self, code: int, body: str, retry_after: Optional[float]):
        super().__init__(_explain(code, body))
        self.code = code
        self.retry_after = retry_after


# The gateway speaks OpenAI's error envelope. Failures a user can act on get a
# sentence naming the next step; anything else surfaces its own message.
_ERROR_HINTS = {
    "insufficient_quota": (
        "Infistar account is out of credit — top up at https://infistar.cc "
        "(余额充值) and run the node again."
    ),
    "invalid_api_key": (
        "Infistar rejected INFISTAR_API_KEY. Check the token in TongFlow "
        "Settings against 令牌管理 in the console."
    ),
    "model_not_found": (
        "This key can't use that model. Pick another entry in the node's model "
        "dropdown — GET /v1/models lists what the key's group allows."
    ),
}


def _explain(status: int, body: str) -> str:
    try:
        obj = json.loads(body)
    except (ValueError, TypeError):
        obj = None
    err = obj.get("error") if isinstance(obj, dict) else None
    if isinstance(obj, dict) and not isinstance(err, dict) and obj.get("message"):
        return f"Infistar rejected the request: {obj['message']}"
    if not isinstance(err, dict):
        return f"HTTP {status} from Infistar: {body[:400]}"
    hint = _ERROR_HINTS.get(str(err.get("code") or ""))
    if hint:
        return hint
    message = str(err.get("message") or "").strip()
    if message:
        field = str(err.get("param") or "").strip()
        return f"Infistar rejected the request{f' ({field})' if field else ''}: {message}"
    return f"HTTP {status} from Infistar: {body[:400]}"


def _headers(content_type: Optional[str] = None) -> Dict[str, str]:
    headers = {"Authorization": f"Bearer {_require_api_key()}"}
    if content_type:
        headers["Content-Type"] = content_type
    return headers


def _http(
    method: str,
    url: str,
    *,
    body: bytes | None = None,
    content_type: Optional[str] = None,
    timeout: float = 300,
) -> Tuple[bytes, str]:
    """Raw request; returns (body, content-type)."""
    log.info("%s %s", method, url)
    req = Request(url, data=body, headers=_headers(content_type), method=method)
    try:
        resp = urlopen(req, timeout=timeout)  # noqa: S310
    except HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
        log.error("HTTP %s on %s\nresponse body: %s", e.code, url, err_body[:2000])
        retry_after: Optional[float] = None
        try:
            raw_ra = e.headers.get("Retry-After") if e.headers else None
            retry_after = float(raw_ra) if raw_ra else None
        except ValueError:
            retry_after = None
        raise _HttpStatus(e.code, err_body or str(e.reason), retry_after) from e
    except URLError as e:
        raise RuntimeError(f"Network error contacting Infistar: {e.reason}") from e
    except TimeoutError as e:
        raise RuntimeError(f"Infistar request timed out after {timeout:.0f}s") from e
    return resp.read(), resp.headers.get_content_type() or ""


def _parse_json(raw: bytes) -> Dict[str, Any]:
    text = raw.decode("utf-8", errors="replace")
    obj = json.loads(text) if text.strip() else {}
    if not isinstance(obj, dict):
        raise RuntimeError(f"Unexpected non-object response: {text[:200]}")
    return obj


def _json_request(
    method: str, url: str, body: Dict[str, Any] | None = None, timeout: float = 300
) -> Dict[str, Any]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    raw, _ = _http(
        method,
        url,
        body=data,
        content_type="application/json" if data else None,
        timeout=timeout,
    )
    return _parse_json(raw)


def _multipart(
    fields: Dict[str, str], files: List[Tuple[str, str, str, bytes]]
) -> Tuple[bytes, str]:
    boundary = "----tongflow" + os.urandom(16).hex()
    line = boundary.encode()
    parts: List[bytes] = []
    for name, value in fields.items():
        parts.append(b"--" + line)
        parts.append(f'Content-Disposition: form-data; name="{name}"'.encode())
        parts.append(b"")
        parts.append(value.encode("utf-8"))
    for field, filename, mime, content in files:
        parts.append(b"--" + line)
        parts.append(
            f'Content-Disposition: form-data; name="{field}"; filename="{filename}"'.encode()
        )
        parts.append(f"Content-Type: {mime}".encode())
        parts.append(b"")
        parts.append(content)
    parts.append(b"--" + line + b"--")
    parts.append(b"")
    return b"\r\n".join(parts), f"multipart/form-data; boundary={boundary}"


def _download(url: str) -> Tuple[bytes, str]:
    try:
        resp = urlopen(url, timeout=600)  # noqa: S310
    except HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} downloading result: {e.reason}") from e
    except URLError as e:
        raise RuntimeError(f"Network error downloading result: {e.reason}") from e
    return resp.read(), resp.headers.get_content_type() or ""


def _asset_bytes(a: Asset) -> bytes:
    return base64.b64decode(a.bytesBase64)


def _data_url(a: Asset, *, default_mime: str) -> str:
    mime = (a.mime or default_mime).strip() or default_mime
    return f"data:{mime};base64,{a.bytesBase64}"


# ── Model resolution (shortlist + the key's own model list) ────────────────

_KEY_MODELS: set[str] | None = None


def _key_model_ids() -> set[str]:
    """Ids the current key may use; empty on failure (never blocks a run)."""
    global _KEY_MODELS
    if _KEY_MODELS is None:
        ids: set[str] = set()
        try:
            obj = _json_request("GET", str(TONGFLOW_MODEL_CATALOG["url"]), timeout=30)
            for rec in obj.get("data") or []:
                mid = rec.get("id") if isinstance(rec, dict) else None
                if isinstance(mid, str) and mid.strip():
                    ids.add(mid.strip())
        except RuntimeError as e:
            log.warning("could not list Infistar models: %s", e)
        _KEY_MODELS = ids
    return _KEY_MODELS


def _active_model(slot: str) -> str:
    models = TONGFLOW_SLOT_MODELS[slot]
    if not _REQUEST_MODEL:
        return models[0]
    if _REQUEST_MODEL in models or _REQUEST_MODEL in _key_model_ids():
        return _REQUEST_MODEL
    raise RuntimeError(
        f"unknown model {_REQUEST_MODEL!r} for {slot} (not in the shortlist and "
        f"not in this key's Infistar model list)"
    )


def _size(width: Optional[int], height: Optional[int], choices: List[str], default: str) -> str:
    """Snap a pixel size to the closest supported `WxH` string."""
    if not width or not height:
        return default
    target = width / height
    best, best_err = default, float("inf")
    for s in choices:
        w, h = s.split("x")
        err = abs(int(w) / int(h) - target)
        if err < best_err:
            best, best_err = s, err
    return best


# ── Chat completions (text + vision) ───────────────────────────────────────


def _sse_text(raw: str) -> str:
    """Concatenate `choices[0].delta.content` over an SSE chat stream."""
    out: List[str] = []
    for line in raw.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            chunk = json.loads(payload)
        except ValueError:
            continue
        choices = chunk.get("choices") if isinstance(chunk, dict) else None
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            continue
        delta = choices[0].get("delta") or choices[0].get("message") or {}
        piece = delta.get("content") if isinstance(delta, dict) else None
        if isinstance(piece, str):
            out.append(piece)
    return "".join(out)


def _chat(slot: str, messages: List[Dict[str, Any]]) -> str:
    """One completion, streamed. Streaming is the default because new-api
    gateways intermittently return an empty non-streamed body for the GPT-5.x
    family while the same request streams fine."""
    body: Dict[str, Any] = {
        "model": _active_model(slot),
        "messages": messages,
        "stream": True,
    }
    temperature = _adv("temperature", 1.0)
    if temperature != 1.0:
        body["temperature"] = temperature
    effort = str(_adv("reasoning_effort", "default"))
    if effort != "default":
        body["reasoning_effort"] = effort

    for attempt in range(CHAT_EMPTY_RETRIES + 1):
        raw, _ = _http(
            "POST",
            f"{_base_url()}/chat/completions",
            body=json.dumps(body).encode("utf-8"),
            content_type="application/json",
            timeout=600,
        )
        text = _sse_text(raw.decode("utf-8", errors="replace")).strip()
        if text:
            return text
        log.warning("empty completion (attempt %d/%d)", attempt + 1, CHAT_EMPTY_RETRIES + 1)
    raise RuntimeError(
        "Infistar returned an empty completion three times — try another model."
    )


def _text_message(*parts: str) -> List[Dict[str, Any]]:
    return [{"role": "user", "content": "\n\n".join(p for p in parts if p)}]


def _vision_message(prompt: str, images: List[Asset]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for img in images:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _data_url(img, default_mime="image/png")},
            }
        )
    return [{"role": "user", "content": content}]


# ── Images ─────────────────────────────────────────────────────────────────


def _image_from_response(obj: Dict[str, Any]) -> Asset:
    """Read the first image out of an OpenAI-shaped images response."""
    data = obj.get("data")
    entry = data[0] if isinstance(data, list) and data and isinstance(data[0], dict) else {}
    b64 = entry.get("b64_json")
    if isinstance(b64, str) and b64.strip():
        return asset(base64.b64decode(b64), mime="image/png")
    url = entry.get("url")
    if isinstance(url, str) and url.startswith("http"):
        content, ctype = _download(url)
        return asset(content, mime=ctype or "image/png")
    raise RuntimeError(f"Infistar image response carried no image: {str(obj)[:300]}")


def _image_generate(slot: str, prompt: str, width: Optional[int], height: Optional[int]) -> Asset:
    body: Dict[str, Any] = {
        "model": _active_model(slot),
        "prompt": prompt,
        "size": _size(width, height, IMAGE_SIZES, DEFAULT_IMAGE_SIZE),
        "n": 1,
    }
    quality = str(_adv("quality", "auto"))
    if quality != "auto":
        body["quality"] = quality
    return _image_from_response(
        _json_request("POST", f"{_base_url()}/images/generations", body, timeout=600)
    )


def _image_edit(slot: str, prompt: str, images: List[Asset], width: Optional[int], height: Optional[int]) -> Asset:
    """`/v1/images/edits` as multipart: several sources go in as `image[]`,
    which is how the OpenAI-compatible route takes multi-reference edits."""
    # No `size`: the edits channels reject every value with "当前模型没有支持
    # 本次媒体规格的可用渠道", the same way the video route does. The edit keeps
    # the source image's geometry, which is what the node wants anyway.
    fields = {"model": _active_model(slot), "prompt": prompt, "n": "1"}
    quality = str(_adv("quality", "auto"))
    if quality != "auto":
        fields["quality"] = quality
    # OpenAI's spelling: one `image` for a single source, `image[]` per source
    # for a multi-reference edit. Which models have a multi-image channel is a
    # separate matter — step-image-edit-2 only routes single-image edits, so
    # image-fusion defaults to gpt-image-2.
    files = [
        (
            "image[]" if len(images) > 1 else "image",
            (img.filename or f"source{i}.png"),
            (img.mime or "image/png"),
            _asset_bytes(img),
        )
        for i, img in enumerate(images)
    ]
    body, ctype = _multipart(fields, files)
    raw, _ = _http(
        "POST", f"{_base_url()}/images/edits", body=body, content_type=ctype, timeout=600
    )
    return _image_from_response(_parse_json(raw))


# ── Videos (OpenAI Videos shape: submit → poll → download) ─────────────────


def _video(slot: str, prompt: str, *, image: Optional[Asset]) -> Asset:
    body: Dict[str, Any] = {"model": _active_model(slot), "prompt": prompt}
    # `seconds` / `size` are deliberately omitted: every combination is
    # rejected with "当前模型没有支持本次媒体规格的可用渠道", while the same
    # request without them is accepted, so the gateway's video channels take
    # the model's own defaults. The env vars are an escape hatch for the day a
    # channel does accept a spec. The node's own duration / width / height are
    # not read at all: `@node_slot` deep-`model_construct`s without filling
    # defaults, so an untouched optional field has no attribute to read.
    spec_seconds = _env("INFISTAR_VIDEO_SECONDS")
    if spec_seconds:
        body["seconds"] = int(spec_seconds)
    spec_size = _env("INFISTAR_VIDEO_SIZE")
    if spec_size:
        body["size"] = spec_size
    if image is not None:
        # `input_reference` is a plain string (a URL or data URI) — passing the
        # OpenAI-style {"image_url": ...} object is rejected by the Go decoder.
        body["input_reference"] = _data_url(image, default_mime="image/png")

    created = _json_request("POST", f"{_base_url()}/videos", body, timeout=300)
    task_id = created.get("id") or created.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise RuntimeError(f"Infistar video submit returned no id: {str(created)[:300]}")
    log.info("submitted video task %s", task_id)

    deadline = time.monotonic() + _poll_timeout()
    delay = POLL_INITIAL_DELAY_S
    while True:
        time.sleep(delay)
        delay = POLL_INTERVAL_S
        try:
            state = _json_request("GET", f"{_base_url()}/videos/{task_id}", timeout=60)
        except _HttpStatus as e:
            if e.code == 429:
                delay = max(POLL_INTERVAL_S, e.retry_after or POLL_INTERVAL_S)
                log.warning("rate limited polling %s; retrying in %.0fs", task_id, delay)
                if time.monotonic() >= deadline:
                    raise
                continue
            raise
        status = str(state.get("status") or "").lower()
        if status == "completed":
            meta = state.get("metadata")
            url = state.get("url") or (meta.get("url") if isinstance(meta, dict) else None)
            if isinstance(url, str) and url.startswith("http"):
                content, ctype = _download(url)
                return asset(content, mime=ctype or "video/mp4")
            raw, ctype = _http(
                "GET", f"{_base_url()}/videos/{task_id}/content", timeout=600
            )
            return asset(raw, mime=ctype or "video/mp4")
        if status in ("failed", "error", "cancelled"):
            err = state.get("error")
            msg = err.get("message") if isinstance(err, dict) else (err or state.get("fail_reason"))
            raise RuntimeError(f"Infistar video task {task_id} {status}: {msg or 'unknown error'}")
        pct = state.get("progress")
        progress(
            f"Generating video ({status or 'queued'})",
            percent=float(pct) if isinstance(pct, (int, float)) else None,
        )
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Infistar video task {task_id} did not finish within "
                f"{int(_poll_timeout())}s (last status: {status or 'unknown'})"
            )


# ── Slots ──────────────────────────────────────────────────────────────────


@node_slot(NodeSlots.GEN_TEXT)
def gen_text(input: GenTextInput) -> GenTextOutput:
    prompt = (input.userPrompt or "").strip()
    answer = _chat(
        "gen-text",
        _text_message(prompt, f"User input: {input.text}", "Output only the requested answer."),
    )
    return GenTextOutput(success=True, text=answer)


@node_slot(NodeSlots.COMBINE_TEXT)
def combine_text(input: CombineTextInput) -> CombineTextOutput:
    prompt = (input.userPrompt or "").strip()
    answer = _chat(
        "combine-text",
        _text_message(prompt, "User input:\n" + "\n\n".join(input.texts), "Output only the requested answer."),
    )
    return CombineTextOutput(success=True, text=answer)


def _parse_split(raw: str) -> List[str]:
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if "\n" in s:
            _, _, s = s.partition("\n")
        s = s.strip().removesuffix("```").strip()
    obj = json.loads(s)
    items = obj if isinstance(obj, list) else (obj.get("texts") if isinstance(obj, dict) else None)
    if not isinstance(items, list) or not all(isinstance(x, str) for x in items):
        raise ValueError("model did not return a JSON array of strings")
    cleaned = [x.strip() for x in items if x.strip()]
    if not cleaned:
        raise ValueError("model returned an empty split")
    return cleaned


@node_slot(NodeSlots.SPLIT_TEXT)
def split_text(input: SplitTextInput) -> SplitTextOutput:
    instruction = (input.userPrompt or "").strip() or "Split into natural, coherent segments."
    raw = _chat(
        "split-text",
        _text_message(
            f"Split the following text according to this instruction:\n{instruction}",
            "Return ONLY a JSON array of strings — no prose, no code fences. "
            "Preserve the original wording; do not summarize.",
            f"TEXT:\n{input.text}",
        ),
    )
    try:
        return SplitTextOutput(success=True, texts=_parse_split(raw))
    except (ValueError, json.JSONDecodeError) as e:
        return SplitTextOutput(success=False, error=str(e))


@node_slot(NodeSlots.IMAGE_GEN_TEXT)
def image_gen_text(input: ImageGenTextInput) -> ImageGenTextOutput:
    prompt = (input.text or "").strip() or "Describe this image."
    messages = _vision_message(prompt, [input.image])
    if input.system:
        messages.insert(0, {"role": "system", "content": input.system})
    answer = _chat("image-gen-text", messages)
    return ImageGenTextOutput(success=True, text=answer)


@node_slot(NodeSlots.IMAGE_GEN)
def image_gen(input: ImageGenInput) -> ImageGenOutput:
    image = _image_generate("image-gen", input.text, input.width, input.height)
    return ImageGenOutput(success=True, image=image)


@node_slot(NodeSlots.IMAGE_EDIT)
def image_edit(input: ImageEditInput) -> ImageEditOutput:
    image = _image_edit("image-edit", input.text, [input.image], input.width, input.height)
    return ImageEditOutput(success=True, image=image)


@node_slot(NodeSlots.IMAGE_FUSION)
def image_fusion(input: ImageFusionInput) -> ImageFusionOutput:
    images = list(input.images or [])
    if not images:
        return ImageFusionOutput(success=False, error="no reference images")
    image = _image_edit("image-fusion", input.text, images, input.width, input.height)
    return ImageFusionOutput(success=True, image=image)


@node_slot(NodeSlots.TEXT_GEN_VIDEO)
def text_gen_video(input: TextGenVideoInput) -> TextGenVideoOutput:
    video = _video("text-gen-video", input.text, image=None)
    return TextGenVideoOutput(success=True, video=video)


@node_slot(NodeSlots.IMAGE_GEN_VIDEO)
def image_gen_video(input: ImageGenVideoInput) -> ImageGenVideoOutput:
    video = _video("image-gen-video", input.text, image=input.image)
    return ImageGenVideoOutput(success=True, video=video)


@node_slot(NodeSlots.TRANSCRIBE)
def transcribe(input: TranscribeInput) -> TranscribeOutput:
    fields = {"model": _active_model("transcribe")}
    if input.language:
        fields["language"] = input.language
    if input.context:
        fields["prompt"] = input.context
    body, ctype = _multipart(
        fields,
        [
            (
                "file",
                input.audio.filename or "audio.mp3",
                input.audio.mime or "audio/mpeg",
                _asset_bytes(input.audio),
            )
        ],
    )
    raw, _ = _http(
        "POST", f"{_base_url()}/audio/transcriptions", body=body, content_type=ctype, timeout=900
    )
    obj = _parse_json(raw)
    text = obj.get("text")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError(f"Infistar transcription returned no text: {str(obj)[:300]}")
    return TranscribeOutput(success=True, text=text.strip())


@node_slot(NodeSlots.ARRANGE_GROUP)
def arrange_group(input: ArrangeGroupInput) -> ArrangeGroupOutput:
    result = arrange_group_output(input.model_dump())
    return ArrangeGroupOutput.model_construct(**result)


@node_slot(NodeSlots.DROP_VIDEO)
def drop_video(input: DropVideoInput) -> DropVideoOutput:
    result = drop_video_output(input.model_dump())
    return DropVideoOutput.model_construct(**result)


# Runtime dispatcher. The @node_slot wrapper takes the raw dict here and dumps
# the BaseModel return back to a dict; `Any` reflects that I/O boundary.
_SLOT_HANDLERS: Dict[str, Any] = {
    NodeSlots.GEN_TEXT: gen_text,
    NodeSlots.SPLIT_TEXT: split_text,
    NodeSlots.COMBINE_TEXT: combine_text,
    NodeSlots.IMAGE_GEN_TEXT: image_gen_text,
    NodeSlots.IMAGE_GEN: image_gen,
    NodeSlots.IMAGE_EDIT: image_edit,
    NodeSlots.IMAGE_FUSION: image_fusion,
    NodeSlots.TEXT_GEN_VIDEO: text_gen_video,
    NodeSlots.IMAGE_GEN_VIDEO: image_gen_video,
    NodeSlots.TRANSCRIBE: transcribe,
    NodeSlots.ARRANGE_GROUP: arrange_group,
    NodeSlots.DROP_VIDEO: drop_video,
}


def _write(out: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(out, ensure_ascii=False))
    sys.stdout.flush()


def main() -> int:
    global _REQUEST_MODEL
    try:
        raw = sys.stdin.read()
        req = json.loads(raw) if raw.strip() else {}
        prompt = req.get("prompt") if isinstance(req, dict) else {}
        if not isinstance(prompt, dict):
            prompt = {}
        slot = str(req.get("nodeSlot") or "") if isinstance(req, dict) else ""
        _REQUEST_MODEL = (
            str(req.get("model") or "").strip() if isinstance(req, dict) else ""
        )

        handler = _SLOT_HANDLERS.get(slot)
        if handler is None:
            raise RuntimeError(f"unsupported nodeSlot: {slot!r}")
        out = handler(prompt)
    except Exception as e:  # noqa: BLE001 — surfaced as ABI failure
        _write({"success": False, "error": str(e)})
        return 1

    _write(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
