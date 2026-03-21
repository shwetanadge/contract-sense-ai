from src.ingestion.pdf_parser import extract_text

# Run parser on GDPR PDF
text = extract_text("data/sample_contracts/gdpr.pdf")

# Print first 500 characters
print("=== FIRST 500 CHARACTERS ===")
print(text[:500])

# Print how many characters total
print(f"\n=== TOTAL CHARACTERS EXTRACTED: {len(text)} ===")

# Print first 3 pages worth of cleaned text
print("\n=== FIRST 3 PAGES OF CLEANED TEXT ===")
print(text[:2000])