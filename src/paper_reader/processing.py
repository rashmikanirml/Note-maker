from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

import pdfplumber
try:
    import pytesseract
except Exception:  # pragma: no cover - optional dependency
    pytesseract = None

try:
    from pdf2image import convert_from_path
except Exception:  # pragma: no cover - optional dependency
    convert_from_path = None
from docx import Document
from pypdf import PdfReader

from paper_reader.ai_provider import get_ai_provider


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def load_document(path: str | Path) -> str:
    document_path = Path(path)
    suffix = document_path.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix}")

    if suffix == ".pdf":
        text = _load_pdf(document_path)
        if not text.strip():
            raise ValueError(
                "No readable text was found in this PDF. If this is a scanned paper, it needs OCR or a text-based PDF."
            )
        return text
    if suffix == ".docx":
        text = _load_docx(document_path)
        if not text.strip():
            raise ValueError("No readable text was found in this DOCX file.")
        return text

    text = document_path.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        raise ValueError("No readable text was found in this file.")
    return text


def _load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    extracted = "\n\n".join(pages).strip()
    if extracted:
        return extracted

    with pdfplumber.open(str(path)) as pdf:
        fallback_pages: list[str] = []
        for page in pdf.pages:
            fallback_pages.append(page.extract_text() or "")
    combined = "\n\n".join(fallback_pages).strip()
    if combined:
        return combined

    # last resort: try OCR (optional system deps)
    ocr_text = _ocr_pdf(path)
    if ocr_text:
        return ocr_text

    return ""


def _ocr_pdf(path: Path) -> str:
    """Attempt OCR on PDF pages using pdf2image + pytesseract.

    Requires system dependencies: Poppler for pdf2image and Tesseract binary for pytesseract.
    Returns combined OCR text or empty string if OCR couldn't run.
    """
    if pytesseract is None or convert_from_path is None:
        return ""

    try:
        images = convert_from_path(str(path), dpi=200)
    except Exception:
        return ""

    texts: list[str] = []
    for img in images:
        try:
            texts.append(pytesseract.image_to_string(img))
        except Exception:
            continue
    return "\n\n".join(texts).strip()


def _load_docx(path: Path) -> str:
    document = Document(str(path))
    return "\n".join(paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()).strip()


def build_summary(text: str, max_sentences: int = 5) -> str:
    sentences = _split_sentences(text)
    if not sentences:
        return "No readable text was found in the document."
    return " ".join(sentences[:max_sentences])


def build_study_notes(text: str, max_notes: int = 8) -> list[str]:
    notes: list[str] = []
    for line in text.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        if len(cleaned) > 100:
            cleaned = cleaned[:97].rstrip() + "..."
        notes.append(f"- {cleaned}")
        if len(notes) >= max_notes:
            break
    if not notes:
        notes.append("- No structured notes could be produced from the input text.")
    return notes


def build_question_answers(text: str, questions: Iterable[str] | None = None) -> list[tuple[str, str]]:
    question_list = list(questions or [
        "What is the main topic of the paper?",
        "What are the key ideas or findings?",
        "What theoretical concepts are important here?",
    ])
    answer_source = build_summary(text, max_sentences=8)
    extracted_keywords = _extract_keywords(text)
    answers: list[tuple[str, str]] = []
    for question in question_list:
        answer = (
            f"{answer_source}\n\n"
            f"Key terms: {', '.join(extracted_keywords[:8]) or 'None detected.'}\n"
            f"Theory focus: explain the underlying concepts, definitions, and implications in the document."
        )
        answers.append((question, answer))
    return answers


def analyze_document(path: str | Path, questions: Iterable[str] | None = None):
    document_path = Path(path)
    text = load_document(document_path)
    provider = get_ai_provider()
    question_list = list(questions or [
        "What is the main topic of the paper?",
        "What are the key ideas or findings?",
        "What theoretical concepts are important here?",
    ])
    summary = provider.summarize(text)
    qas = provider.answer_questions(text, question_list)
    notes = provider.build_notes(text)

    provider_name = None
    # resilient provider stores last_provider_name, otherwise try provider.name
    provider_name = getattr(provider, "last_provider_name", None) or getattr(provider, "name", None) or provider.__class__.__name__

    return {
        "title": document_path.stem,
        "source_path": str(document_path),
        "extracted_text": text,
        "summary": summary,
        "questions_and_answers": qas,
        "study_notes": notes,
        "provider": provider_name,
    }


def _split_sentences(text: str) -> list[str]:
    candidates = re.split(r"(?<=[.!?])\s+", text.replace("\r", " "))
    return [candidate.strip() for candidate in candidates if candidate.strip()]


def _extract_keywords(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", text.lower())
    frequencies: dict[str, int] = {}
    for token in tokens:
        frequencies[token] = frequencies.get(token, 0) + 1
    return [word for word, _ in sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))]
