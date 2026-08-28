from ocr_pdf import process, print_report

SRC_PDF = "raw/Delhi Right to Information Act 2001.pdf"
OUT_TXT = "processed/delhi_rti_2001_ocr_final.txt"

if __name__ == "__main__":
    out, cleaned, flags = process(SRC_PDF, OUT_TXT, label="Delhi RTI Act, 2001")
    if out:
        print_report(cleaned, flags)