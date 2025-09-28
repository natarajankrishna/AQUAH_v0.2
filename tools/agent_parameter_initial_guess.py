"""
Multi-provider CREST first-guess pipeline (patched + Gemini key fix)
------------------------------------------------------------------
Step-1  describe_basin_for_crest()  – Vision LLM (OpenAI / Claude‑3 / Gemini / DeepSeek)
Step-2  estimate_crest_args()       – CrewAI agent infers full CREST parameter set
"""

from __future__ import annotations
import base64
import io
import json
import mimetypes
import os
import re
import types
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT PATCH: make LiteLLM pick up the same key google.generativeai uses
# ─────────────────────────────────────────────────────────────────────────────
# If the user only exports GOOGLE_API_KEY (works for google‑generativeai), set a
# fallback alias so LiteLLM (expects GEMINI_API_KEY) can authenticate.
if "GEMINI_API_KEY" not in os.environ and "GOOGLE_API_KEY" in os.environ:
    os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]

# ╭──────────────────────── Helper: identify provider from model name ─────────╮

def _detect_provider(model: str) -> str:
    """Return a canonical provider key based on model name prefix."""
    if model.startswith(("anthropic/", "claude-")):
        return "anthropic"
    if model.startswith(("gemini", "models/", "gemini/")):
        return "gemini"
    if model.startswith("deepseek"):
        return "deepseek"
    return "openai"

# ╰────────────────────────────────────────────────────────────────────────────╯

# ╭────────────────────── Helper: image → Base64 (Claude limit safe) ─────────╮
_CLAUDE_B64_LIMIT = 5 * 1024 * 1024  # 5 MB (Anthropic counts b64 size)
_CLAUDE_BIN_LIMIT = int(_CLAUDE_B64_LIMIT * 0.75)  # ≈3.9 MB raw bytes


def _jpeg_downsize(data: bytes, *, limit: int) -> Tuple[str, bytes]:
    """Iteratively down-scale & JPEG‑recompress until size ≤ *limit* bytes."""
    with Image.open(io.BytesIO(data)) as im:
        im = im.convert("RGB")
        w, h = im.size
        qualities = [85, 75, 65, 55]
        while True:
            for q in qualities:
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=q, optimize=True)
                out = buf.getvalue()
                if len(out) <= limit:
                    return "image/jpeg", out
            if w < 450:  # minimum width safety net
                return "image/jpeg", out
            w, h = int(w * 0.85), int(h * 0.85)
            im = im.resize((w, h), Image.LANCZOS)


def _file_to_b64(path: str, provider: str) -> Tuple[str, str]:
    """Return (mime, base64_string) for *path*; compress if provider == Claude."""
    data = Path(path).read_bytes()
    mime = mimetypes.guess_type(path)[0] or "image/png"
    if provider == "anthropic" and len(data) > _CLAUDE_BIN_LIMIT:
        mime, data = _jpeg_downsize(data, limit=_CLAUDE_BIN_LIMIT)
    return mime, base64.b64encode(data).decode()

# ╰────────────────────────────────────────────────────────────────────────────╯

# ╭────────────────────────── Provider-specific chat wrappers ─────────────────╮


def _call_openai(model: str, messages: list, *, temperature: float):
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return (
        client.chat.completions.create(
            model=model, messages=messages
        )
        .choices[0]
        .message.content
    )


def _call_claude(model: str, messages: list, *, temperature: float):
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return client.messages.create(
        model=model, messages=messages, temperature=temperature, max_tokens=1024
    ).content[0].text



def _call_gemini(model: str, parts, *, temperature: float):
    import google.generativeai as genai

    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    return genai.GenerativeModel(model).generate_content(
        parts, generation_config={"temperature": temperature}
    ).text


def _call_deepseek(model: str, messages: list, *, temperature: float):
    url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1/chat/completions")
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
        },
        timeout=60,
    ).json()
    return resp["choices"][0]["message"]["content"]


# ╰────────────────────────────────────────────────────────────────────────────╯

# ╭──────────────────────────── Universal vision_chat() ───────────────────────╮

def vision_chat(
    model: str, images: List[str], user_text: str, *, temperature: float = 0.3
) -> str:
    provider = _detect_provider(model)

    # OpenAI / DeepSeek: image_url objects
    if provider in {"openai", "deepseek"}:
        msg = [{"role": "user", "content": []}]
        for p in images:
            mime, b64 = _file_to_b64(p, provider)
            msg[0]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                }
            )
        msg[0]["content"].append({"type": "text", "text": user_text})
        fn = _call_openai if provider == "openai" else _call_deepseek
        return fn(model, msg, temperature=temperature)

    # Anthropic Claude‑3: block format
    if provider == "anthropic":
        blocks = []
        for p in images:
            mime, b64 = _file_to_b64(p, provider)
            blocks.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": mime, "data": b64},
                }
            )
        blocks.append({"type": "text", "text": user_text})
        return _call_claude(
            model, [{"role": "user", "content": blocks}], temperature=temperature
        )

    # Google Gemini: list[ text, PIL.Image … ]
    parts = [user_text] + [Image.open(p) for p in images]
    return _call_gemini(model, parts, temperature=temperature)


# ╰────────────────────────────────────────────────────────────────────────────╯

# ╭─────────────────────── STEP‑1  Basin description (Vision) ─────────────────╮

def describe_basin_for_crest(
    args: Any, *, vision_model: str = "gpt-4o", temperature: float = 0.3
) -> str:
    img_dir = Path(args.figure_path)
    imgs = [img_dir / "basin_map_with_gauges.png", img_dir / "basic_data.png"]

    prompt = (
        f"Basin Name: {args.basin_name}\n"
        f"Basin Area (km2): {args.basin_area}\n"
        f"Selected Outlet Gauge ID: {args.gauge_id}\n\n"
        "Write ONE paragraph (≤8 sentences, 200‑300 words) summarising all "
        "physiographic & hydrologic traits relevant to initial CREST parameters. "
        "List location, elevation range, slope, land‑cover, imperviousness, soil "
        "texture / Ks class, reservoirs, drainage density, snow influence, "
        "flashiness.  Use numeric values when visible; if uncertain, qualify "
        "with 'likely'.  Plain text only."
    )

    return vision_chat(str(vision_model), [str(p) for p in imgs], prompt, temperature=temperature).strip()


# ╰────────────────────────────────────────────────────────────────────────────╯

# ╭────────────────── Helper: CrewAI‑native LLM object by provider ────────────╮

from crewai import LLM


def _ensure_gemini_route(name: str) -> str:
    """Prepends the gemini/ route if missing so LiteLLM can infer provider."""
    return name if name.startswith("gemini/") else f"gemini/{name}"


def _get_crewai_llm(model_name: str, *, temperature: float = 0) -> LLM:
    """Return a CrewAI‑compatible LLM wrapper while avoiding `provider` leaks
    to endpoints that reject unknown parameters (OpenAI, Anthropic)."""
    provider = _detect_provider(model_name)

    # 1) Gemini ─ ensure "gemini/" prefix, rely on LiteLLM auto‑routing
    if provider == "gemini":
        model_name = _ensure_gemini_route(model_name)
        return LLM(model=model_name, temperature=temperature)

    # 2) OpenAI – DO NOT pass provider (causes 400)
    if provider == "openai":
        return LLM(model=model_name, temperature=temperature)

    # 3) Anthropic – same, DO NOT pass provider to avoid "provider: extra inputs"
    if provider == "anthropic":
        return LLM(model=model_name, temperature=temperature)

    # 4) DeepSeek – needs explicit provider flag
    if provider == "deepseek":
        return LLM(model=model_name, provider="deepseek", temperature=temperature)

    # 5) Anything else -> error
    raise ValueError(f"Unsupported provider for model '{model_name}'")(model=model_name, temperature=temperature)


# ╰────────────────────────────────────────────────────────────────────────────╯

# ╭──────────────────────────── STEP‑2  CREST parameters ──────────────────────╮

from crewai import Agent, Crew, Task, Process
from crewai_tools import PDFSearchTool, WebsiteSearchTool

class SafeWebsiteSearchTool(WebsiteSearchTool):
    """Wrapper that swallows underlying HTTP errors and returns a plain string."""

    def run(self, *, search_query: str):  # type: ignore[override]
        try:
            return super().run(search_query=search_query)
        except Exception as e:  # noqa: BLE001
            return f"WEBSITE_SEARCH_ERROR: {e}"


def estimate_crest_args(basin_desc: str, args: Any) -> Tuple[str, str]:
    """Return (python_code_line, justification) via CrewAI agent."""

    pdf_tool = PDFSearchTool(pdf="EF5_tools/EF5_Parameters.pdf")
    site_tool = SafeWebsiteSearchTool(website="https://chrimerss.github.io/EF5/docs/")

    llm = _get_crewai_llm(args.llm_model_name, temperature=0)

    estimator = Agent(
        role="CREST Parameter Estimator",
        goal="Generate reasonable first‑guess CREST parameters from EF5 docs",
        backstory="Seasoned hydrologist who calibrates CREST/EF5.",
        tools=[pdf_tool, site_tool],
        llm=llm,
        verbose=True,
    )

    # (1) Retrieve default parameter ranges
    guide_task = Task(
        description=(
            "Use tools to summarise default CREST parameter ranges and physical "
            "meaning (wm, b, im, ke, fc, iwu, under, leaki, th, isu, alpha, "
            "beta, alpha0) within 500 words."
        ),
        expected_output="≤500 words summary",
        agent=estimator,
    )
    Crew(agents=[estimator], tasks=[guide_task], process=Process.sequential, verbose=True).kickoff()
    guide = guide_task.output.raw.strip()

    # (2) Produce JSON line with code + explanation
    prompt = (
        # "You are a hydrologist. Using the parameter guide and basin description, also this simulation is for a Hurricane (The hurricane result shows a sharp peak — it rises quickly and falls just as fast), "
        # "propose first-guess CREST parameters, use parameters that can lead to higher peak discharge.\n\n"
        "You are a hydrologist. Using the parameter guide and basin description, "
        "propose first-guess CREST parameters\n\n"
        f"Basin description:\n{basin_desc}\n\n"
        f"Parameter guide:\n{guide}\n\n"
        "Return EXACTLY one line of JSON, formatted like (change the values to the ones you think are reasonable):\n"
        "{\"code\":\"crest_args = types.SimpleNamespace(wm=200.0, b=10.0, im=0.15, ke=0.8, "
        "fc=50.0, iwu=25.0, under=1.0, leaki=0.05, th=100.0, isu=0.0, alpha=1.5, beta=0.6, "
        "alpha0=1.0, grid_on=False)\","
        # "{\"code\":\"crest_args = types.SimpleNamespace(wm=<value>, b=<value>, im=<value>, ke=<value>, "
        # "fc=<value>, iwu=<value>, under=<value>, leaki=<value>, th=<value>, isu=<value>, alpha=<value>, beta=<value>, "
        # "alpha0=<value>, grid_on=False)\","
        "\"explanation\":\"each param justified in 100 to 300 words\"}\n"
        "NO markdown, NO extra keys."
    )

    est_task = Task(description=prompt, expected_output="One-line JSON", agent=estimator)
    Crew(agents=[estimator], tasks=[est_task], process=Process.sequential, verbose=True).kickoff()

    raw = est_task.output.raw.strip()
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        raise ValueError("LLM did not return valid JSON:\n" + raw)

    reply = json.loads(match.group())
    print("=== LLM used for parameter estimation ===\n", estimator.llm.model)
    return reply["code"], reply["explanation"]


# ╰────────────────────────────────────────────────────────────────────────────╯

# ╭────────────────────────── Pipeline entry: get_initial_crest_args ──────────╮

def get_initial_crest_args(args):
    """
    *args* must expose attributes:
      basin_name, basin_area, gauge_id, figure_path
      llm_model_name   – text‑only model for CREST estimation
      vision_model_name (optional) – vision model for STEP‑1
    Returns (crest_args_namespace, explanation_text)
    """

    vision_model = getattr(
        args, "vision_model_name", getattr(args, "llm_model_name", "gpt-4o")
    )

    # 1. Vision step
    basin_desc = describe_basin_for_crest(args, vision_model=vision_model, temperature=0.3)
    print("\n--- Basin Description ---\n", basin_desc)

    # 2. Parameter estimation
    code_line, explanation = estimate_crest_args(basin_desc, args)

    # 3. Execute generated code to obtain SimpleNamespace
    ns: Dict[str, Any] = {"types": types}
    exec(code_line, ns)
    crest_args = ns["crest_args"]

    # 4. Log & return
    print("\n=== Generated crest_args ===")
    for k, v in vars(crest_args).items():
        print(f"{k:>8}: {v}")
    print("\n=== Agent explanation ===\n", explanation)
    return crest_args, explanation


# ╰─────────────────────────────────────────────────────────────────────────────╯
