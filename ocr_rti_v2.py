from ocr_pdf import process, print_report

SRC_PDF = "raw/02.The RTI Act, 2005_As on 01022011_English.pdf"
OUT_TXT = "processed/rti_act_2005_v2_ocr_final.txt"

if __name__ == "__main__":
    out, cleaned, flags = process(SRC_PDF, OUT_TXT, label="RTI Act, 2005 (v2 scanned copy)")
    if out:
        print_report(cleaned, flags)