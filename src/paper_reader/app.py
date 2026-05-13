from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from paper_reader.pdf_export import export_report
from paper_reader.processing import analyze_document, extract_questions_from_text, load_document
import json
from pathlib import Path
import os


class PaperReaderApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Paper Reader - Answer Questions from Papers")
        self.root.geometry("1400x960")
        self.analysis: dict | None = None
        self.source_path: Path | None = None
        self.user_questions: list[str] = []
        self.extracted_text: str = ""

        container = ttk.Frame(self.root, padding=12)
        container.pack(fill="both", expand=True)

        self.status_var = tk.StringVar(value="1. Open a paper  →  2. Enter questions  →  3. Get answers")
        ttk.Label(container, textvariable=self.status_var, font=("Arial", 10, "bold")).pack(anchor="w")

        controls = ttk.Frame(container)
        controls.pack(fill="x", pady=(10, 12))
        ttk.Button(controls, text="📄 Open Paper", command=self.open_document).pack(side="left", padx=(0, 5))
        ttk.Button(controls, text="📝 Load Questions", command=self.load_questions).pack(side="left", padx=(0, 5))
        ttk.Button(controls, text="🤖 Get Answers", command=self.get_answers).pack(side="left", padx=(0, 5))
        ttk.Button(controls, text="💾 Export PDF", command=self.export_pdf).pack(side="left", padx=(0, 5))
        ttk.Button(controls, text="⚙️ Settings", command=self.open_settings).pack(side="right")

        # Left side: Questions
        left_panel = ttk.Frame(container)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 6))

        ttk.Label(left_panel, text="Questions (one per line):", font=("Arial", 9, "bold")).pack(anchor="w")
        self.questions_box = scrolledtext.ScrolledText(left_panel, wrap=tk.WORD, height=15)
        self.questions_box.pack(fill="both", expand=True, pady=(0, 8))

        # Right side: Answers and info
        right_panel = ttk.Frame(container)
        right_panel.pack(side="right", fill="both", expand=True)

        self.qa_box = self._add_section(right_panel, "AI Answers", 15)
        self.status_detail_var = tk.StringVar(value="No paper loaded yet.")
        ttk.Label(right_panel, textvariable=self.status_detail_var, font=("Arial", 8)).pack(anchor="w", pady=(4, 0))

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
            self.extracted_text = load_document(path)
            self.source_path = Path(path)

            extracted_questions = extract_questions_from_text(self.extracted_text)
            self.questions_box.delete("1.0", tk.END)
            if extracted_questions:
                self.questions_box.insert(tk.END, "\n".join(extracted_questions))
                self.user_questions = extracted_questions
                self.status_detail_var.set(
                    f"Loaded {self.source_path.name}. Found {len(extracted_questions)} question(s). Generating answers..."
                )
                self.root.update_idletasks()

                self.analysis = analyze_document(self.source_path, questions=extracted_questions)
                provider = self.analysis.get("provider", "Unknown")
                self.status_detail_var.set(
                    f"Loaded {self.source_path.name}. Found {len(extracted_questions)} question(s). Answers ready (Provider: {provider})."
                )
                self._render_answers()
            else:
                self.status_detail_var.set(
                    f"Loaded {self.source_path.name}. No questions detected. You can type questions and click Get Answers."
                )
                self._set_text(self.qa_box, "No questions were detected in the paper. Enter questions and click Get Answers.")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Could not read file", str(exc))
            self.status_detail_var.set(f"Error loading paper: {str(exc)[:80]}")

    def load_questions(self) -> None:
        path = filedialog.askopenfilename(
            title="Load questions from file",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            self.questions_box.delete("1.0", tk.END)
            self.questions_box.insert(tk.END, content)
            self.status_detail_var.set(f"✅ Questions loaded from {Path(path).name}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Could not load questions", str(exc))

    def open_settings(self) -> None:
        dlg = tk.Toplevel(self.root)
        dlg.title("Settings")
        dlg.geometry("640x220")
        dlg.transient(self.root)

        frm = ttk.Frame(dlg, padding=12)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="Hugging Face Token (HF_TOKEN)").pack(anchor="w")
        token_entry = ttk.Entry(frm, width=80)
        token_entry.pack(fill="x", pady=(0, 8))

        ttk.Label(frm, text="Hugging Face Model (HF_MODEL) (optional)").pack(anchor="w")
        model_entry = ttk.Entry(frm, width=80)
        model_entry.pack(fill="x", pady=(0, 8))

        # Load existing config if present
        try:
            cfg_path = Path.home() / ".note_maker_config.json"
            if cfg_path.exists():
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                token_entry.insert(0, cfg.get("HF_TOKEN", ""))
                model_entry.insert(0, cfg.get("HF_MODEL", ""))
        except Exception:
            pass

        def save_settings() -> None:
            token = token_entry.get().strip()
            model = model_entry.get().strip()
            cfg = {}
            try:
                cfg_path = Path.home() / ".note_maker_config.json"
                if cfg_path.exists():
                    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                cfg = {}
            if token:
                cfg["HF_TOKEN"] = token
            else:
                cfg.pop("HF_TOKEN", None)
            if model:
                cfg["HF_MODEL"] = model
            else:
                cfg.pop("HF_MODEL", None)

            try:
                cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
                # Also set in current process so changes take effect immediately
                if token:
                    os.environ["HF_TOKEN"] = token
                else:
                    os.environ.pop("HF_TOKEN", None)
                if model:
                    os.environ["HF_MODEL"] = model
                else:
                    os.environ.pop("HF_MODEL", None)
                self.status_detail_var.set("Settings saved.")
                dlg.destroy()
            except Exception as exc:
                messagebox.showerror("Could not save settings", str(exc))

        btn_frame = ttk.Frame(frm)
        btn_frame.pack(fill="x", pady=(10, 0))
        ttk.Button(btn_frame, text="Save", command=save_settings).pack(side="right")

    def get_answers(self) -> None:
        if not self.source_path or not self.extracted_text:
            messagebox.showwarning("No paper loaded", "Please open a paper first.")
            return

        questions_text = self.questions_box.get("1.0", tk.END).strip()
        if not questions_text:
            self.user_questions = extract_questions_from_text(self.extracted_text)
            if self.user_questions:
                self.questions_box.delete("1.0", tk.END)
                self.questions_box.insert(tk.END, "\n".join(self.user_questions))
            else:
                messagebox.showwarning("No questions", "No questions found in the paper. Please enter questions (one per line).")
                return

        self.user_questions = [q.strip() for q in self.questions_box.get("1.0", tk.END).split("\n") if q.strip()]
        
        try:
            self.status_detail_var.set(f"🔄 Analyzing paper with {len(self.user_questions)} question(s)...")
            self.root.update()
            
            self.analysis = analyze_document(self.source_path, questions=self.user_questions)
            provider = self.analysis.get("provider", "Unknown")
            self.status_detail_var.set(f"✅ Analysis complete! Provider: {provider}")
            self._render_answers()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Analysis failed", str(exc))
            self.status_detail_var.set(f"❌ Error: {str(exc)[:60]}")

    def _render_answers(self) -> None:
        assert self.analysis is not None
        qa_text = "\n\n".join(
            f"Q: {question}\n{'─' * 80}\nA: {answer}"
            for question, answer in self.analysis["questions_and_answers"]
        )
        self._set_text(self.qa_box, qa_text)

    def export_pdf(self) -> None:
        if not self.analysis or not self.source_path:
            messagebox.showinfo("Nothing to export", "Get answers first by clicking 'Get Answers'.")
            return

        suggested_name = f"{self.source_path.stem}-answers.pdf"
        destination = filedialog.asksaveasfilename(
            title="Save PDF report",
            defaultextension=".pdf",
            initialfile=suggested_name,
            filetypes=[("PDF Files", "*.pdf")],
        )
        if not destination:
            return

        try:
            # Create a text version of Q&A for export
            qa_text = "\n\n".join(
                f"Q: {q}\n\nA: {a}"
                for q, a in self.analysis["questions_and_answers"]
            )
            export_report(
                destination,
                f"{self.source_path.stem} - Q&A Report",
                "",  # summary
                self.analysis["questions_and_answers"],
                [],  # study notes
            )
            messagebox.showinfo("Export complete", f"Saved PDF report to {destination}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Could not export PDF", str(exc))

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
