from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol, Iterable

import requests


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
        from paper_reader.processing import build_question_answers

        return build_question_answers(text, questions)

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
    if os.getenv("HF_TOKEN"):
        providers.append(HuggingFaceInferenceProvider())
    if os.getenv("OPENAI_API_KEY"):
        providers.append(OpenAIProvider())
    # always append local fallback last
    providers.append(LocalFallbackProvider())

    if len(providers) == 1:
        return providers[0]
    return ResilientProvider(providers=providers)
