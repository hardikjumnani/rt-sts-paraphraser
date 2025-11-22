import os
from dotenv import load_dotenv
import requests
from typing import Optional

load_dotenv()


# Default to the HF router chat endpoint; you can override via env if needed
HF_API_URL = os.getenv("HF_API_URL", "https://router.huggingface.co/v1/chat/completions")

# Must be set in your environment
HF_API_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")


class ParaphraserError(Exception):
    pass


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _vocab_descriptor(vocab_level: float) -> str:
    v = _clip01(vocab_level)
    if v < 0.33:
        return "very simple, everyday vocabulary, as if explaining to a 10-year-old"
    elif v < 0.66:
        return "normal, conversational vocabulary for a general adult audience"
    else:
        return "advanced, domain-specific vocabulary appropriate for experts"


def _strength_descriptor(strength: float) -> str:
    s = _clip01(strength)
    if s < 0.33:
        return "only light rewording, staying as close as possible to the original phrasing"
    elif s < 0.66:
        return "moderate rephrasing that changes wording and structure but keeps the same meaning"
    else:
        return "strong paraphrasing that significantly changes wording and structure while strictly preserving meaning"


def _formality_descriptor(formality: float) -> str:
    f = _clip01(formality)
    if f < 0.33:
        return "very informal, conversational tone with contractions and relaxed phrasing"
    elif f < 0.66:
        return "neutral tone, suitable for a broad audience"
    else:
        return "formal, professional tone with precise wording and no slang"


def _build_user_message(text: str, vocab: float, strength: float, formality: float) -> str:
    vocab_desc = _vocab_descriptor(vocab)
    strength_desc = _strength_descriptor(strength)
    formality_desc = _formality_descriptor(formality)

    # This becomes the "user" message in the chat API
    user_msg = f"""
Rewrite the following text according to these controls:

- Vocabulary level: {vocab_desc}
- Paraphrase strength: {strength_desc}
- Formality: {formality_desc}

Guidelines:
- Preserve the original meaning exactly.
- Do not add new facts.
- Output only the paraphrased text, without quotes and without explanations.
- Keep the language in English.

Original text:
{text}

Paraphrased text:
""".strip()

    return user_msg


def _call_hf_api(user_message: str, max_new_tokens: int = 256) -> str:
    if not HF_API_URL or not HF_API_TOKEN or not HF_MODEL:
        raise ParaphraserError(
            "HF_API_URL, HF_TOKEN (HF_API_TOKEN), and HF_MODEL environment variables must be set."
        )

    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json",
    }

    # OpenAI-style /v1/chat/completions payload
    payload = {
        "model": HF_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a precise English paraphrasing assistant. "
                    "You strictly follow style controls and preserve meaning."
                ),
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
        "temperature": 0.3,
        "top_p": 1.0,
        "max_tokens": max_new_tokens,
        # Many HF backends support this seed field for determinism
        "seed": 42,
    }

    resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise ParaphraserError(
            f"Inference API error {resp.status_code}: {resp.text[:400]}"
        )

    data = resp.json()

    try:
        content = data["choices"][0]["message"]["content"]
    except Exception as e:
        raise ParaphraserError(
            f"Unexpected response format: {data}"
        ) from e

    # Router returns a string content for plain text use cases
    return content.strip()


def paraphrase(
    text: str,
    vocab: float,
    strength: float,
    formality: float,
    max_new_tokens: Optional[int] = None,
) -> str:
    """
    Paraphrase text using a free LLM via Hugging Face router (chat/completions API).

    :param text: Input text to paraphrase.
    :param vocab: Vocabulary level in [0,1].
    :param strength: Paraphrase strength in [0,1].
    :param formality: Formality level in [0,1].
    :param max_new_tokens: Optional override for output length.
    """
    if not text or not text.strip():
        raise ValueError("text must be a non-empty string")

    user_message = _build_user_message(text.strip(), vocab, strength, formality)
    # Rough heuristic to avoid truncation
    mnt = max_new_tokens or max(64, int(len(text.split()) * 1.5))
    output = _call_hf_api(user_message, max_new_tokens=mnt)
    return output
