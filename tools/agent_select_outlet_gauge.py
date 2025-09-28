"""Universal *select_outlet_gauge_by_image* — supports vision chat across
OpenAI, Anthropic Claude‑3, Google Gemini Vision, and OpenAI‑compatible DeepSeek.

`model_name` switches provider in one line. Handles Claude's **5 MB (Base64) limit**:
* Actual compression threshold set to 3.9 MB (binary), which with Base64's ~33% overhead stays under 5 MB.
* Iteratively scales down to min 450px width and gradually reduces JPEG quality (85 → 70 → 60) to ensure compliance.

Returns `(gauge_id, raw_llm_response)`.
"""
from __future__ import annotations

import base64
import io
import mimetypes
import os
import re
from pathlib import Path
from typing import List, Tuple

import requests  # DeepSeek / HTTP
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CLAUDE_LIMIT_B64 = 5 * 1024 * 1024  # 5 MB (Anthropic counts Base64 bytes)
CLAUDE_BIN_LIMIT = int(CLAUDE_LIMIT_B64 * 0.75)  # ≈3.9 MB binary


def _encode_b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _compress_image(data: bytes, *, bin_limit: int) -> tuple[str, bytes]:
    """JPEG‑compress *data* until ≤bin_limit. Return (mime, data)."""
    with Image.open(io.BytesIO(data)) as im:
        im = im.convert("RGB")
        width, height = im.size
        quality_seq = [85, 75, 65, 55]
        while True:
            for q in quality_seq:
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=q, optimize=True)
                new_data = buf.getvalue()
                if len(new_data) <= bin_limit:
                    return "image/jpeg", new_data
            # Still too big → reduce by 15%
            if width < 450:
                # Already at minimum size, can't compress further => return minimized result
                return "image/jpeg", new_data
            width = int(width * 0.85)
            height = int(height * 0.85)
            im = im.resize((width, height), Image.LANCZOS)


def _file_to_b64(path: str, *, provider: str) -> tuple[str, str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    data = p.read_bytes()
    mime = mimetypes.guess_type(p.name)[0] or "image/png"

    if provider == "anthropic":
        if len(data) > CLAUDE_BIN_LIMIT:
            mime, data = _compress_image(data, bin_limit=CLAUDE_BIN_LIMIT)
    # Others: no compression needed
    return mime, _encode_b64(data)


def _find_digits(text: str) -> str:
    # Find all digit sequences of length 6-10
    matches = re.findall(r"\b(\d{6,10})\b", text)
    if not matches:
        return text.strip()
    
    # Count occurrences of each match
    counts = {}
    for m in matches:
        counts[m] = counts.get(m, 0) + 1
    
    # Return the most frequent match
    return max(counts.items(), key=lambda x: x[1])[0]

# ---------------------------------------------------------------------------
# Provider APIs (unchanged)
# ---------------------------------------------------------------------------

def _call_openai(model, messages, *, api_key, temperature):
    from openai import OpenAI
    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    return client.chat.completions.create(model=model, messages=messages).choices[0].message.content

def _call_claude(model, messages, *, api_key, temperature, max_tokens=1024):
    import anthropic
    client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
    return client.messages.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens).content[0].text

from PIL import Image
def _call_gemini(model: str, parts, api_key=None, temperature=0.1):
    import google.generativeai as genai
    genai.configure(api_key=api_key or os.getenv("GOOGLE_API_KEY"))
    return genai.GenerativeModel(model).generate_content(
        parts, generation_config={"temperature": temperature}
    ).text

def _call_deepseek(model, messages, *, api_key, temperature):
    api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
    url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1/chat/completions")
    resp = requests.post(url, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, json={"model": model, "messages": messages, "temperature": temperature}, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def select_outlet_gauge_by_image(*, input_text: str, gauges_description: str, image_path: str, basic_data_image_path: str | None = None, facc_image_path: str | None = None, model_name: str = "gpt-4o", temperature: float = 0.3,) -> Tuple[str, str]:
    img_paths: List[str] = [image_path]
    if basic_data_image_path:
        img_paths.append(basic_data_image_path)
    if facc_image_path:
        img_paths.append(facc_image_path)

    SYSTEM = (
        "You are a hydrologist who can interpret maps and select the most appropriate USGS gauge to represent the **natural** basin outlet.\n"
        "The user supplies: (1) a watershed boundary (yellow), (2) a base map with DEM/elevation, and (3) a flow-accumulation raster with gauges (blue triangles). "
        "The flow-accumulation image is the primary evidence for stream convergence; the DEM shows elevation and highlights reservoirs/lakes.\n\n"
        "Apply the following ordered rules when selecting ONE gauge (earlier rules override later ones):\n"
        "0) If the user's text clearly mentions or implies a specific gauge, city, or location, select that gauge with highest priority. This overrides all other rules.\n"
        "1) **Absolute exclusion:** Disqualify any gauge located downstream of, or directly on the outlet of, a reservoir/lake (visible as a large blue polygon in the base map with gauges, "
        "and corresponding to flat, low-value areas in the DEM). This rule applies only if rule 0 doesn't apply.\n"
        "2) From the remaining gauges, prefer the one at the lowest elevation point on the basin boundary where flow naturally exits the watershed (use DEM).\n"
        "3) Prefer gauges capturing the largest drainage area and the highest flow-accumulation values, indicating convergence of most tributaries (use the flow-accumulation raster).\n"
        "4) Prefer gauges with extensive, reliable USGS discharge records.\n"
        "5) **Second verification:** Before responding, re-check that the chosen gauge is upstream of all reservoirs/lakes and sits at a natural outlet (low elevation, high accumulation). "
        "If not, discard it and re-evaluate.\n\n"
        "Return your response in this format:\nSelected gauge: [gauge ID number]\nExplanation: [brief explanation of your choice]"
    )

    USER_BLOCK = f"User input: '{input_text}'\n\nGauge info:\n{gauges_description}\n\nPlease select the most appropriate outlet gauge following the rules above."

    if model_name.startswith(("anthropic/", "claude-")):
        provider = "anthropic"
    elif model_name.startswith(("gemini", "models/")):
        provider = "gemini"
    elif model_name.startswith("deepseek"):
        provider = "deepseek"
    else:
        provider = "openai"

    if provider in {"openai", "deepseek"}:
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": []}]
        for p in img_paths:
            mime, b64 = _file_to_b64(p, provider=provider)
            messages[1]["content"].append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
        messages[1]["content"].append({"type": "text", "text": USER_BLOCK})
        raw = (_call_openai if provider == "openai" else _call_deepseek)(model_name, messages, api_key=None, temperature=temperature)

    elif provider == "anthropic":
        blocks = []
        for p in img_paths:
            mime, b64 = _file_to_b64(p, provider=provider)
            blocks.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}})
        blocks.append({"type": "text", "text": USER_BLOCK})
        raw = _call_claude(model_name, [{"role": "user", "content": blocks}], api_key=None, temperature=temperature)

    else:  # gemini
        # parts = [USER_BLOCK] + [Path(p).read_bytes() for p in img_paths]
        parts = [USER_BLOCK] + [Image.open(p) for p in img_paths]
        raw = _call_gemini(model_name, parts, api_key=None, temperature=temperature)

    gauge_id = _find_digits(raw)
    print(f"=== Model used for gauge selection ===\n{model_name}")
    return gauge_id, raw

def select_outlet_gauge(args):
    figure_path = args.figure_path
    basin_map_png = os.path.join(figure_path, "combined_maps.png")
    basic_data_png = os.path.join(figure_path, "basic_data.png")
    facc_png = os.path.join(figure_path, "facc_with_gauges.png")
    gauge_id, raw = select_outlet_gauge_by_image(input_text=args.input_text, gauges_description=args.gauges_description, image_path=basin_map_png, basic_data_image_path=basic_data_png, facc_image_path=facc_png, model_name=args.llm_model_name, temperature=0.3)
    return gauge_id, raw
