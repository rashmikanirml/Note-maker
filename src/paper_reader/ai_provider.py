from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Protocol, Iterable

import requests
from pathlib import Path


class AIProvider(Protocol):
    def summarize(self, text: str) -> str: ...

    def answer_questions(self, text: str, questions: list[str]) -> list[tuple[str, str]]: ...

    def build_notes(self, text: str) -> list[str]: ...


@dataclass(slots=True)
class LocalFallbackProvider:
    """A simple local provider used when no external API is configured."""

    def summarize(self, text: str) -> str:
        from paper_reader.processing import build_summary
        return build_summary(text)

    @property
    def name(self) -> str:
        return "Local"

    def answer_questions(self, text: str, questions: list[str]) -> list[tuple[str, str]]:
        """Answer questions by extracting relevant sentences from the text."""
        answers: list[tuple[str, str]] = []
        for question in questions:
            answer = self._find_answer(text, question)
            answers.append((question, answer))
        return answers
    
    def _find_answer(self, text: str, question: str) -> str:
        """Build a grounded answer from the most relevant paragraphs in the paper."""
        import re

        stopwords = {
            "what", "why", "how", "when", "where", "which", "who", "the", "and", "for", "with", "from",
            "that", "this", "into", "about", "their", "there", "these", "those", "explain", "describe",
            "define", "discuss", "compare", "state", "list", "name", "question",
        }

        terms = [w for w in re.findall(r"\b[a-z]{3,}\b", question.lower()) if w not in stopwords]
        if not terms:
            terms = re.findall(r"\b[a-z]{3,}\b", question.lower())

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
        if not paragraphs:
            paragraphs = [text.strip()]

        scored: list[tuple[int, str]] = []
        for paragraph in paragraphs:
            lower = paragraph.lower()
            score = 0
            for term in terms:
                if term in lower:
                    score += lower.count(term)
            if score > 0:
                scored.append((score, paragraph))

        if scored:
            scored.sort(key=lambda item: item[0], reverse=True)
            best_paragraphs = [item[1] for item in scored[:2]]
            sentences: list[str] = []
            for paragraph in best_paragraphs:
                for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
                    cleaned = sentence.strip()
                    if cleaned:
                        sentences.append(cleaned)
                    if len(sentences) >= 4:
                        break
                if len(sentences) >= 4:
                    break

            answer = " ".join(sentences)
            if len(answer) > 700:
                answer = answer[:697].rstrip() + "..."
            return answer

        from paper_reader.processing import build_summary

        return build_summary(text, max_sentences=3)

    def build_notes(self, text: str) -> list[str]:
        from paper_reader.processing import build_study_notes
        return build_study_notes(text)


@dataclass(slots=True)
class HuggingFaceInferenceProvider:
    model: str = "google/flan-t5-base"
    token: str | None = None

    def __post_init__(self) -> None:
        self.token = self.token or os.getenv("HF_TOKEN")

    def summarize(self, text: str) -> str:
        prompt = f"Summarize this paper in 5 concise sentences:\n\n{text[:4000]}"
        return self._generate(prompt)

    @property
    def name(self) -> str:
        return "HuggingFace"

    def answer_questions(self, text: str, questions: list[str]) -> list[tuple[str, str]]:
        answers: list[tuple[str, str]] = []
        for question in questions:
            prompt = (
                f"Answer the question using the paper, with theoretical detail and clarity.\n"
                f"Question: {question}\n\nPaper:\n{text[:4000]}"
            )
            answers.append((question, self._generate(prompt)))
        return answers

    def build_notes(self, text: str) -> list[str]:
        prompt = f"Extract 8 study notes from this paper as short bullet points:\n\n{text[:4000]}"
        response = self._generate(prompt)
        return [f"- {line.strip()}" for line in response.splitlines() if line.strip()]

    def _generate(self, prompt: str) -> str:
        if not self.token:
            raise RuntimeError("HF_TOKEN is not configured")

        response = requests.post(
            f"https://api-inference.huggingface.co/models/{self.model}",
            headers={"Authorization": f"Bearer {self.token}"},
            json={"inputs": prompt},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, list) and payload:
            first = payload[0]
            if isinstance(first, dict):
                return first.get("generated_text") or first.get("summary_text") or str(first)
        if isinstance(payload, dict):
            return payload.get("generated_text") or payload.get("summary_text") or str(payload)
        return str(payload)


@dataclass(slots=True)
class OpenAIProvider:
    """Simple OpenAI provider using REST calls. Requires OPENAI_API_KEY env var."""
    model: str = "gpt-3.5-turbo"
    token: str | None = None

    def __post_init__(self) -> None:
        self.token = self.token or os.getenv("OPENAI_API_KEY")

    def summarize(self, text: str) -> str:
        prompt = f"Summarize this paper in 5 concise sentences:\n\n{text[:6000]}"
        return self._chat_completion(prompt)

    @property
    def name(self) -> str:
        return "OpenAI"

    def answer_questions(self, text: str, questions: list[str]) -> list[tuple[str, str]]:
        answers: list[tuple[str, str]] = []
        for question in questions:
            prompt = (
                f"You are a helpful assistant. Use the following paper to answer the question with theoretical depth.\n\nPaper:\n{text[:6000]}\n\nQuestion: {question}"
            )
            answers.append((question, self._chat_completion(prompt)))
        return answers

    def build_notes(self, text: str) -> list[str]:
        prompt = f"List 8 concise study notes from the paper:\n\n{text[:6000]}"
        response = self._chat_completion(prompt)
        return [f"- {line.strip()}" for line in response.splitlines() if line.strip()]

    def _chat_completion(self, prompt: str) -> str:
        if not self.token:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 800,
        }
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        resp = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        if choices:
            message = choices[0].get("message", {})
            return message.get("content", "")
        return ""


@dataclass(slots=True)
class ResilientProvider:
    providers: list[AIProvider]
    # name of the provider that last successfully produced output
    last_provider_name: str | None = None

    def summarize(self, text: str) -> str:
        last_exc: Exception | None = None
        for p in self.providers:
            try:
                result = p.summarize(text)
                # record which provider succeeded
                self.last_provider_name = getattr(p, "name", p.__class__.__name__)
                return result
            except Exception as exc:  # try next provider
                last_exc = exc
        raise RuntimeError("All AI providers failed") from last_exc

    def answer_questions(self, text: str, questions: list[str]) -> list[tuple[str, str]]:
        last_exc: Exception | None = None
        for p in self.providers:
            try:
                result = p.answer_questions(text, questions)
                self.last_provider_name = getattr(p, "name", p.__class__.__name__)
                return result
            except Exception as exc:
                last_exc = exc
        raise RuntimeError("All AI providers failed to answer questions") from last_exc

    def build_notes(self, text: str) -> list[str]:
        last_exc: Exception | None = None
        for p in self.providers:
            try:
                result = p.build_notes(text)
                self.last_provider_name = getattr(p, "name", p.__class__.__name__)
                return result
            except Exception as exc:
                last_exc = exc
        raise RuntimeError("All AI providers failed to build notes") from last_exc


def get_ai_provider() -> AIProvider:
    providers: list[AIProvider] = []
    # prefer configured external providers
    # Prefer HF token from env, otherwise from user config file
    hf_token = os.getenv("HF_TOKEN")
    hf_model_env = os.getenv("HF_MODEL")
    if not hf_token:
        try:
            cfg_path = Path.home() / ".note_maker_config.json"
            if cfg_path.exists():
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                hf_token = cfg.get("HF_TOKEN") or hf_token
                hf_model_env = hf_model_env or cfg.get("HF_MODEL")
        except Exception:
            # ignore config read errors
            hf_token = hf_token

    if hf_token:
        providers.append(HuggingFaceInferenceProvider(token=hf_token, model=hf_model_env or None))

    if os.getenv("OPENAI_API_KEY"):
        providers.append(OpenAIProvider())
    # Google Gemini / Generative Models
    if os.getenv("GEMINI_API_KEY"):
        # lazy-import provider class defined below
        providers.append(GeminiProvider())
    # always append local fallback last
    providers.append(LocalFallbackProvider())

    if len(providers) == 1:
        return providers[0]
    return ResilientProvider(providers=providers)


@dataclass(slots=True)
class GeminiProvider:
    """Simple connector for Google Generative Models (Gemini) REST API.

    Requires `GEMINI_API_KEY` environment variable (Bearer token or API key).
    Set `GEMINI_MODEL` to e.g. `models/text-bison-001` or `models/chat-bison-001`.
    """
    model: str = "models/text-bison-001"
    api_key: str | None = None

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.getenv("GEMINI_API_KEY")
        self.model = os.getenv("GEMINI_MODEL", self.model)

    @property
    def name(self) -> str:
        return "Gemini"

    def _call_generate(self, prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured")

        url = f"https://generativelanguage.googleapis.com/v1beta2/{self.model}:generateText"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "prompt": {"text": prompt},
            "temperature": 0.2,
            "maxOutputTokens": 800,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # Google generative responses commonly include 'candidates' with 'output'
        candidates = data.get("candidates") or []
        if candidates:
            first = candidates[0]
            return first.get("output") or first.get("content") or str(first)
        # fallback for other shapes
        return data.get("output") or data.get("result", "") or str(data)

    def summarize(self, text: str) -> str:
        prompt = f"Summarize this paper in 5 concise sentences:\n\n{text[:8000]}"
        return self._call_generate(prompt)

    def answer_questions(self, text: str, questions: list[str]) -> list[tuple[str, str]]:
        answers: list[tuple[str, str]] = []
        for question in questions:
            prompt = (
                f"You are a helpful assistant. Use the following paper to answer the question concisely and accurately.\n\nPaper:\n{text[:8000]}\n\nQuestion: {question}\n\nAnswer:")
            resp = self._call_generate(prompt)
            answers.append((question, resp))
        return answers

    def build_notes(self, text: str) -> list[str]:
        prompt = f"List 8 concise study notes from the paper as short bullet points:\n\n{text[:8000]}"
        resp = self._call_generate(prompt)
        return [f"- {line.strip()}" for line in resp.splitlines() if line.strip()]
