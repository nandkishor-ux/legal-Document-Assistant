import re

from text_cleaning import is_noise, clean_line_list, collapse_blanks

raw_path = "processed/rti_act_2005_raw.txt"
output_path = "processed/rti_act_2005_clean.txt"

with open(raw_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

lines = raw_text.splitlines()

kept_lines, n_dropped = clean_line_list(lines)
dropped = [ln for ln in lines if is_noise(ln)]

final = collapse_blanks(kept_lines)

cleaned_text = "\n".join(final).rstrip() + "\n"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(cleaned_text)

# --- report ---
print(f"Lines read: {len(lines)}")
print(f"Lines removed as noise: {len(dropped)}")
print(f"Lines kept: {len(final)}")
print(f"Raw chars: {len(raw_text)}  ->  Cleaned chars: {len(cleaned_text)}")
print("=" * 80)
print("Sample of removed lines (first 30):")
for i, ln in enumerate(dropped[:30], 1):
    print(f"  {ln!r}")

print("=" * 80)
print("Section-header lines detected in the CLEANED text:")
header_re = re.compile(r"^\s*(\d{1,2})\.\s*[A-Z][a-z]+")
headers = [ln.strip() for ln in final if header_re.match(ln)]
print(f"  {len(headers)} matches")
for h in headers[:45]:
    print(f"    | {h}")

print("=" * 80)
print("First 2000 characters of cleaned text:")
print("=" * 80)
print(cleaned_text[:2000])