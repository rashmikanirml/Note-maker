from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from reportlab.pdfgen import canvas

from paper_reader.pdf_export import export_report
from paper_reader.processing import analyze_document, build_summary, extract_questions_from_text


class ProcessingTests(unittest.TestCase):
    def test_build_summary_uses_first_sentences(self) -> None:
        summary = build_summary("First sentence. Second sentence. Third sentence.")
        self.assertTrue(summary.startswith("First sentence."))


    def test_analyze_document_and_export_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source = temp_path / "sample.txt"
            source.write_text(
                "Introduction to testing.\n\nThis paper explains the core theoretical model and findings.\n",
                encoding="utf-8",
            )

            with mock.patch("paper_reader.processing.get_ai_provider") as mocked_provider_factory:
                mocked_provider = mock.Mock()
                mocked_provider.summarize.return_value = "Summary sentence."
                mocked_provider.answer_questions.return_value = [("Q1", "A1")]
                mocked_provider.build_notes.return_value = ["- Note 1"]
                mocked_provider_factory.return_value = mocked_provider

                analysis = analyze_document(source)

            self.assertEqual(analysis["title"], "sample")
            self.assertTrue(analysis["summary"])
            self.assertTrue(analysis["questions_and_answers"])

            pdf_path = temp_path / "report.pdf"
            exported = export_report(
                pdf_path,
                analysis["title"],
                analysis["summary"],
                analysis["questions_and_answers"],
                analysis["study_notes"],
            )

            self.assertEqual(exported, pdf_path)
            self.assertTrue(exported.exists())
            self.assertGreater(exported.stat().st_size, 0)

    def test_analyze_document_reads_pdf_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            pdf_path = temp_path / "paper.pdf"

            pdf_canvas = canvas.Canvas(str(pdf_path))
            pdf_canvas.drawString(72, 720, "Paper title")
            pdf_canvas.drawString(72, 700, "This paper explains the key theory and findings.")
            pdf_canvas.save()

            with mock.patch("paper_reader.processing.get_ai_provider") as mocked_provider_factory:
                mocked_provider = mock.Mock()
                mocked_provider.summarize.return_value = "Summary sentence."
                mocked_provider.answer_questions.return_value = [("Q1", "A1")]
                mocked_provider.build_notes.return_value = ["- Note 1"]
                mocked_provider_factory.return_value = mocked_provider

                analysis = analyze_document(pdf_path)

            self.assertIn("Paper title", analysis["extracted_text"])
            self.assertTrue(analysis["summary"])
            self.assertTrue(analysis["questions_and_answers"])

    def test_extract_questions_from_text_marks_and_numbered(self) -> None:
        text = (
            "Section A\n"
            "1. Explain supervised learning\n"
            "2) Define precision and recall\n"
            "What is overfitting?\n"
            "This line is not a question\n"
        )

        questions = extract_questions_from_text(text)

        self.assertIn("Explain supervised learning?", questions)
        self.assertIn("Define precision and recall?", questions)
        self.assertIn("What is overfitting?", questions)

    def test_extract_questions_from_text_question_label_and_subparts(self) -> None:
        text = (
            "Question 1: Discuss the role of data preprocessing\n"
            "(a) Compare precision and recall\n"
            "(5 marks)\n"
            "Conclusion paragraph\n"
        )

        questions = extract_questions_from_text(text)

        self.assertIn("Discuss the role of data preprocessing?", questions)
        self.assertIn("Compare precision and recall?", questions)
