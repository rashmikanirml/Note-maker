from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class DocumentAnalysis:
    title: str
    source_path: str
    extracted_text: str
    summary: str
    questions_and_answers: list[tuple[str, str]] = field(default_factory=list)
    study_notes: list[str] = field(default_factory=list)
