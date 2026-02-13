from pdf_utils import extract_text_from_pdf

file_path = "../data/sample.pdf"

text = extract_text_from_pdf(file_path)

print("----- Extracted Text -----\n")
print(text[:500])  # print first 500 characters only
