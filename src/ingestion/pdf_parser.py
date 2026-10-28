import pdfplumber
import re
import os

def extract_text(pdf_path):
    """
    Reads a PDF file and returns cleaned text as a string.
    """
    raw_text = ""

    # Open the PDF and extract text page by page
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                raw_text += page_text + "\n"

    # Clean the extracted text
    cleaned_text = clean_text(raw_text)

    return cleaned_text


def clean_text(text):
    """
    Cleans raw PDF text by fixing common issues. 
    """
    # Fix broken hyphenation (words split across lines)
    text = re.sub(r'-\n', '', text)

    # Remove extra blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    # Remove website footer lines that leak into the text
    lines = [line for line in lines if "gesetze-im-internet.de" not in line]

    # Remove very short lines (page numbers, headers, footers),
    # but keep article/section headings such as "Article 37" or "§ 257"
    heading = re.compile(r"^(Article \d+[a-z]?|§ ?\d+[a-z]?)$")
    lines = [line for line in lines if len(line) > 10 or heading.match(line)]

    # Join everything back together
    cleaned = '\n'.join(lines)

    return cleaned