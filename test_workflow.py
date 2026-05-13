#!/usr/bin/env python
"""Quick test to verify the paper reader works end-to-end."""

from pathlib import Path
from paper_reader.processing import analyze_document

# Test with the sample paper and questions
sample_paper = Path("sample_paper.txt")
sample_questions_file = Path("sample_questions.txt")

if not sample_paper.exists():
    print(f"❌ Sample paper not found: {sample_paper}")
    exit(1)

if not sample_questions_file.exists():
    print(f"❌ Sample questions file not found: {sample_questions_file}")
    exit(1)

# Load questions
with open(sample_questions_file) as f:
    questions = [q.strip() for q in f.readlines() if q.strip()]

print(f"📄 Testing with: {sample_paper.name}")
print(f"❓ Questions to answer: {len(questions)}")
print("-" * 80)

try:
    # Analyze the document
    result = analyze_document(sample_paper, questions=questions)
    
    print(f"\n✅ Analysis successful!")
    print(f"📊 Provider used: {result['provider']}")
    print(f"📝 Title: {result['title']}")
    print(f"📚 Text length: {len(result['extracted_text'])} characters")
    print(f"❓ Generated {len(result['questions_and_answers'])} Q&A pairs\n")
    
    # Display answers
    for i, (q, a) in enumerate(result["questions_and_answers"], 1):
        print(f"\n[Question {i}]")
        print(f"Q: {q}")
        print(f"A: {a[:200]}..." if len(a) > 200 else f"A: {a}")
        print("-" * 80)
    
    print("\n✅ SUCCESS! The app is working correctly.")
    print("You can now open the GUI with: python -m paper_reader")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
