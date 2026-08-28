import pdfplumber

pdf_path = "raw/RTI-Act_English.pdf"
output_path = "processed/rti_act_2005_raw.txt"

full_text = ""
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        text = page.extract_text() or ""
        full_text += text + "\n"

with open(output_path, "w", encoding="utf-8") as f:
    f.write(full_text)

print(f"Total characters extracted: {len(full_text)}")
print("=" * 80)
print("First 1000 characters:")
print("=" * 80)
print(full_text[:1000])