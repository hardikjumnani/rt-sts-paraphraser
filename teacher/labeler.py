from dotenv import load_dotenv
import os
import re
import requests
from functools import lru_cache
from typing import Tuple, Optional

import textstat
from sentence_transformers import SentenceTransformer, util

load_dotenv()
HF_API_URL = os.getenv("HF_API_URL")
HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL")

class LabelerError(Exception):
    pass

class Labeler:
    """
    Computes control scores for text and text pairs:
      - vocab level in [0,1]
      - paraphrase strength in [0,1] (unnormalized or normalized later)
      - formality in [0,1]
    """

    def __init__(self, sbert_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Load SBERT once
        self.sbert = SentenceTransformer(sbert_model_name)

    # -------- Vocabulary level --------

    @staticmethod
    def _type_token_ratio(text: str) -> float:
        tokens = [t.lower() for t in re.findall(r"\w+", text)]
        if not tokens:
            return 0.0
        types = set(tokens)
        return len(types) / len(tokens)

    @staticmethod
    def _normalize(value: float, vmin: float, vmax: float, invert: bool = False) -> float:
        if vmax <= vmin:
            return 0.0
        x = (value - vmin) / (vmax - vmin)
        x = max(0.0, min(1.0, x))
        return 1.0 - x if invert else x

    def vocab_level(self, text: str) -> float:
        text = text.strip()
        if not text:
            return 0.0

        # Flesch-Kincaid grade: higher => more complex
        try:
            fk_grade = textstat.flesch_kincaid_grade(text)
        except Exception:
            fk_grade = 8.0

        # Dale-Chall: higher => more difficult, typical range about 4–15
        try:
            dc_score = textstat.dale_chall_readability_score(text)
        except Exception:
            dc_score = 8.0

        # Type-token ratio
        ttr = self._type_token_ratio(text)

        # Heuristic normalization # treat ~primary school as 0, ~academic as 1
        fk_norm = self._normalize(fk_grade, vmin=2.0, vmax=14.0, invert=False)
        # Dale-Chall: easy ≈5, hard ≈12
        dc_norm = self._normalize(dc_score, vmin=5.0, vmax=12.0, invert=False)
        # TTR: short/boring ≈0.25, diverse ≈0.7
        ttr_norm = self._normalize(ttr, vmin=0.25, vmax=0.7, invert=False)

        base = (fk_norm + dc_norm + ttr_norm) / 3.0

        # 2) penalize very short texts (they often look artificially complex)
        tokens = re.findall(r"\w+", text)
        length_factor = min(1.0, len(tokens) / 12.0)  # <12 tokens push downward
        base *= length_factor

        # 3) mild nonlinearity to spread low values
        vocab = base ** 1.2

        return max(0.0, min(1.0, vocab))

    # -------- Paraphrase strength --------

    @lru_cache(maxsize=100000)
    def _embed(self, text: str):
        return self.sbert.encode(text, convert_to_tensor=True, show_progress_bar=False)

    def paraphrase_strength_raw(self, src_text: str, tgt_text: str) -> float:
        if not src_text.strip() or not tgt_text.strip():
            return 0.0

        e1 = self._embed(src_text.strip())
        e2 = self._embed(tgt_text.strip())
        sim = float(util.cos_sim(e1, e2).item())
        # clamp similarity to a sane range
        sim = max(-1.0, min(1.0, sim))
        strength = 1.0 - sim  # higher similarity => lower strength
        # this is a raw value; you will min-max scale across the dataset later
        return strength

    @staticmethod
    def normalize_strength(strength: float, global_min: float, global_max: float) -> float:
        if global_max <= global_min:
            return 0.0
        x = (strength - global_min) / (global_max - global_min)
        return max(0.0, min(1.0, x))

    # -------- Formality --------

    def formality(self, text: str) -> float:
        text = text.strip()
        if not text:
            return 0.0

        if not HF_API_URL or not HF_TOKEN:
            # Fallback: crude heuristic using textstat formal score if you want
            # For now, just return 0.5 as "neutral"
            return 0.5

        prompt = f"""
You are a style analyst.

Rate the formality of the following English text on a continuous scale
from 0.0 to 1.0, where:

- 0.0 = extremely informal (slang, emojis, text messages, very casual)
- 0.5 = neutral / everyday writing (simple emails, blog posts, basic explanations)
- 1.0 = very formal (academic articles, legal documents, official reports)

Text:
{text}

Respond with only a single number between 0.0 and 1.0, using a dot as decimal separator and no extra words.
""".strip()

        raw = _call_hf_api(prompt, max_new_tokens=8)
        # Extract the first floating-point number from the response
        m = re.search(r"\d+(\.\d+)?", raw)
        if not m:
            return 0.5
        val = float(m.group(0))
        # In case the model returns 0–100 instead, you could detect and rescale, but
        # for now just clip to [0,1].
        if val > 1.0:
            val = val / 100.0 if val <= 100.0 else 1.0
        
        # stretch away from 0.5 to increase contrast
        delta = val - 0.5
        stretched = 0.5 + 1.5 * delta  # 1.5 is a tunable factor
        formality = max(0.0, min(1.0, stretched))

        return formality

    # -------- Convenience wrapper --------

    def compute_controls(self, src_text: str, tgt_text: str) -> Tuple[float, float, float]:
        """
        Compute (vocab_level, paraphrase_strength_raw, formality) for a pair.
        Note: paraphrase_strength is NOT globally normalized here yet.
        """
        vocab = self.vocab_level(tgt_text)
        strength_raw = self.paraphrase_strength_raw(src_text, tgt_text)
        form = self.formality(tgt_text)
        return vocab, strength_raw, form

def _call_hf_api(user_message: str, max_new_tokens: int = 8) -> str:
    if not HF_API_URL or not HF_TOKEN or not HF_MODEL:
        raise Exception(
            "HF_API_URL, HF_TOKEN (HF_API_TOKEN), and HF_MODEL environment variables must be set."
        )

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }

    # OpenAI-style /v1/chat/completions payload
    payload = {
        "model": HF_MODEL,
        "messages": [
            # {
            #     "role": "system",
            #     "content": (
            #         "You are a precise English paraphrasing assistant. "
            #         "You strictly follow style controls and preserve meaning."
            #     ),
            # },
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
        raise LabelerError(
            f"Inference API error {resp.status_code}: {resp.text[:400]}"
        )

    data = resp.json()

    try:
        content = data["choices"][0]["message"]["content"]
    except Exception as e:
        raise LabelerError(
            f"Unexpected response format: {data}"
        ) from e

    # Router returns a string content for plain text use cases
    return content.strip()