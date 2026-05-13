from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from paper_reader.pdf_export import export_report
from paper_reader.processing import analyze_document


class PaperReaderApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Paper Reader Desktop")
        self.root.geometry("1200x860")
        self.analysis: dict | None = None
        self.source_path: Path | None = None

        container = ttk.Frame(self.root, padding=12)
        container.pack(fill="both", expand=True)

        self.status_var = tk.StringVar(value="Load a paper to begin.")
        ttk.Label(container, textvariable=self.status_var).pack(anchor="w")

        controls = ttk.Frame(container)
        controls.pack(fill="x", pady=(10, 12))
        ttk.Button(controls, text="Open Paper", command=self.open_document).pack(side="left")
        ttk.Button(controls, text="Export PDF", command=self.export_pdf).pack(side="left", padx=(8, 0))

        self.summary_box = self._add_section(container, "Summary", 6)
        self.qa_box = self._add_section(container, "Question Answers", 8)
        self.notes_box = self._add_section(container, "Study Notes", 6)
        self.text_box = self._add_section(container, "Extracted Text", 12)

    def _add_section(self, parent: ttk.Frame, title: str, height: int) -> scrolledtext.ScrolledText:
        ttk.Label(parent, text=title).pack(anchor="w")
        widget = scrolledtext.ScrolledText(parent, wrap=tk.WORD, height=height)
        widget.pack(fill="both", expand=True, pady=(0, 10))
        widget.configure(state="disabled")
        return widget

    def open_document(self) -> None:
        path = filedialog.askopenfilename(
            title="Open paper",
            filetypes=[("Documents", "*.pdf *.docx *.txt *.md")],
        )
        if not path:
            return

        try:
            self.analysis = analyze_document(path)
            self.source_path = Path(path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Could not read file", str(exc))
            return

        self._render_analysis()

    def export_pdf(self) -> None:
        if not self.analysis or not self.source_path:
            messagebox.showinfo("Nothing to export", "Open a document first.")
            return

        suggested_name = f"{self.source_path.stem}-report.pdf"
        destination = filedialog.asksaveasfilename(
            title="Save PDF report",
            defaultextension=".pdf",
            initialfile=suggested_name,
            filetypes=[("PDF Files", "*.pdf")],
        )
        if not destination:
            return

        try:
            export_report(
                destination,
                self.analysis["title"],
                self.analysis["summary"],
                self.analysis["questions_and_answers"],
                self.analysis["study_notes"],
            )
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Could not export PDF", str(exc))
            return

        messagebox.showinfo("Export complete", f"Saved PDF report to {destination}")

    def _render_analysis(self) -> None:
        assert self.analysis is not None
        provider = self.analysis.get("provider")
        provider_part = f" — Provider: {provider}" if provider else ""
        self.status_var.set(f"Loaded: {self.analysis['title']}{provider_part}")
        self._set_text(self.summary_box, self.analysis["summary"])
        self._set_text(
            self.qa_box,
            "\n\n".join(
                f"Q: {question}\nA: {answer}" for question, answer in self.analysis["questions_and_answers"]
            ),
        )
        self._set_text(self.notes_box, "\n".join(self.analysis["study_notes"]))
        self._set_text(self.text_box, self.analysis["extracted_text"])

    @staticmethod
    def _set_text(widget: scrolledtext.ScrolledText, content: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, content)
        widget.configure(state="disabled")

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    PaperReaderApp().run()


if __name__ == "__main__":
    main()
