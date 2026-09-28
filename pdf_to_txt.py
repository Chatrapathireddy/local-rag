
import sys
from pathlib import Path

def extract_pdf_text(pdf_path: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    pages_text = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text)
    return "\n\n".join(pages_text)


def main():
    if len(sys.argv) != 3:
        print("Usage: python pdf_to_txt.py <input.pdf> <output.txt>")
        sys.exit(1)

    pdf_path, txt_path = sys.argv[1], sys.argv[2]
    Path(txt_path).parent.mkdir(parents=True, exist_ok=True)

    text = extract_pdf_text(pdf_path)
    Path(txt_path).write_text(text, encoding="utf-8")
    print(f"Extracted {len(text)} characters from {pdf_path} -> {txt_path}")


if __name__ == "__main__":
    main()