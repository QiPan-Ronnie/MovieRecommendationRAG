"""
Explanation generator: given retrieved evidence, produce a natural language
explanation for why a movie is recommended to a user.

Supports two modes:
  1. prompt_only: LLM generates explanation without retrieved evidence
  2. rag: LLM generates explanation grounded in retrieved evidence
  3. kg_only / retrieval_only: specialized evidence-grounded ablations

LLM backend options:
  - HuggingFace transformers (local): Llama 2 7B, Qwen, etc.
  - OpenAI-compatible API (vLLM, Ollama, etc.)
"""
import json
import os
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

RAG_SYSTEM_PROMPT = """You are a movie recommendation assistant. Your task is to explain why a specific movie is recommended to a user based on their viewing history.

RULES:
1. Your explanation MUST be grounded in the provided evidence passages.
2. You MUST cite evidence by referencing specific details from the passages (e.g., "As noted in the film's description...").
3. Do NOT fabricate facts not supported by the evidence.
4. Keep your explanation concise (2-4 sentences).
5. Connect the recommended movie to the user's preferences based on the evidence."""

RAG_USER_TEMPLATE = """## User's Favorite Movies
{history_section}

## Recommended Movie
{candidate_title}

## Retrieved Evidence
{evidence_section}

## Task
Based on the evidence above, explain why "{candidate_title}" is a good recommendation for this user. Cite specific evidence in your explanation."""

PROMPT_ONLY_SYSTEM_PROMPT = """You are a movie recommendation assistant. Your task is to explain why a specific movie is recommended to a user based on their viewing history.

RULES:
1. Keep your explanation concise (2-4 sentences).
2. Connect the recommended movie to the user's preferences.
3. Be specific about what aspects of the movie match the user's taste."""

PROMPT_ONLY_USER_TEMPLATE = """## User's Favorite Movies
{history_section}

## Recommended Movie
{candidate_title} ({candidate_genres})

## Task
Explain why "{candidate_title}" is a good recommendation for this user."""


EVIDENCE_GROUNDED_MODES = {"rag", "hybrid", "kg_only", "retrieval_only"}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExplanationRequest:
    """Input to the explanation generator."""
    user_id: int
    candidate_movie_id: int
    candidate_title: str
    candidate_genres: str
    history_titles: list[str]
    evidence: list[dict] = field(default_factory=list)  # from retriever


@dataclass
class ExplanationResult:
    """Output from the explanation generator."""
    user_id: int
    movie_id: int
    movie_title: str
    explanation: str
    mode: str  # "rag", "prompt_only", "kg_only", "retrieval_only"
    evidence_used: list[dict] = field(default_factory=list)
    raw_prompt: str = ""


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_rag_prompt(request: ExplanationRequest) -> tuple[str, str]:
    """Build system + user prompt for RAG-based explanation."""
    history_section = "\n".join(
        f"- {t}" for t in request.history_titles[:10])

    evidence_section = ""
    for i, ev in enumerate(request.evidence, 1):
        evidence_section += f"[{i}] {ev['text']}\n"

    user_msg = RAG_USER_TEMPLATE.format(
        history_section=history_section,
        candidate_title=request.candidate_title,
        evidence_section=evidence_section.strip(),
    )
    return RAG_SYSTEM_PROMPT, user_msg


def build_prompt_only(request: ExplanationRequest) -> tuple[str, str]:
    """Build system + user prompt for prompt-only explanation (no evidence)."""
    history_section = "\n".join(
        f"- {t}" for t in request.history_titles[:10])

    user_msg = PROMPT_ONLY_USER_TEMPLATE.format(
        history_section=history_section,
        candidate_title=request.candidate_title,
        candidate_genres=request.candidate_genres.replace("|", ", "),
    )
    return PROMPT_ONLY_SYSTEM_PROMPT, user_msg


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------

class LLMBackend:
    """Base class for LLM backends."""
    def generate(self, system_prompt: str, user_prompt: str,
                 max_new_tokens: int = 256) -> str:
        raise NotImplementedError


class HuggingFaceLLM(LLMBackend):
    """Local HuggingFace transformers model (e.g., Qwen2.5, Llama 3)."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                 device: str = "auto", torch_dtype: str = "auto"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        resolved_dtype = dtype_map.get(torch_dtype, "auto")

        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=device, torch_dtype=resolved_dtype,
            trust_remote_code=True)
        self.model.eval()
        print(f"Model loaded: {model_name}")

    def generate(self, system_prompt: str, user_prompt: str,
                 max_new_tokens: int = 256) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(
            self.model.device)

        import torch
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        # Decode only the generated portion
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class APIBackend(LLMBackend):
    """OpenAI-compatible API backend (vLLM, Ollama, text-generation-inference)."""

    def __init__(self, base_url: str = "http://localhost:8000/v1",
                 model: str = "default", api_key: str = "not-needed"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    def generate(self, system_prompt: str, user_prompt: str,
                 max_new_tokens: int = 256) -> str:
        import requests

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": max_new_tokens,
                "temperature": 0.7,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Explanation generator
# ---------------------------------------------------------------------------

class ExplanationGenerator:
    """Generate explanations for recommended movies."""

    def __init__(self, llm: LLMBackend):
        self.llm = llm

    def generate_explanation(self, request: ExplanationRequest,
                             mode: str = "rag") -> ExplanationResult:
        """
        Generate a single explanation.

        Args:
            request: ExplanationRequest with user info + evidence
            mode: "rag", "prompt_only", "kg_only", or "retrieval_only"
        """
        if mode in EVIDENCE_GROUNDED_MODES:
            system_prompt, user_prompt = build_rag_prompt(request)
        else:
            system_prompt, user_prompt = build_prompt_only(request)

        explanation = self.llm.generate(system_prompt, user_prompt)

        return ExplanationResult(
            user_id=request.user_id,
            movie_id=request.candidate_movie_id,
            movie_title=request.candidate_title,
            explanation=explanation,
            mode=mode,
            evidence_used=request.evidence if mode in EVIDENCE_GROUNDED_MODES else [],
            raw_prompt=user_prompt,
        )

    def generate_batch(self, requests: list[ExplanationRequest],
                       mode: str = "rag",
                       output_path: str = None) -> list[ExplanationResult]:
        """Generate explanations for a batch of requests."""
        from tqdm import tqdm

        results = []
        for req in tqdm(requests, desc=f"Generating ({mode})"):
            result = self.generate_explanation(req, mode=mode)
            results.append(result)

        if output_path:
            save_explanations(results, output_path)

        return results


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_explanations(results: list[ExplanationResult], path: str):
    """Save explanation results to JSONL."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            record = {
                "user_id": r.user_id,
                "movie_id": r.movie_id,
                "movie_title": r.movie_title,
                "explanation": r.explanation,
                "mode": r.mode,
                "evidence_used": r.evidence_used,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(results)} explanations -> {path}")


def load_explanations(path: str) -> list[dict]:
    """Load explanation results from JSONL."""
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line))
    return results
