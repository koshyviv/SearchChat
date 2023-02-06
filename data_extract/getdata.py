import os
import PyPDF2

def extract_text_from_pdf(pdf_file):
    with open(pdf_file, 'rb') as file:
        pdf = PyPDF2.PdfReader(file)
        text = ''
        for page in range(len(pdf.pages)):
            text += pdf.pages[page].extract_text()
        return text

def pdf_clean_text(text):
    # add any text cleaning logic here
    return text

def consolidate_pdf_text_to_txt(dir_path, output_file):
    text = ''
    for filename in os.listdir(dir_path):
        if filename.endswith('.pdf'):
            pdf_file = os.path.join(dir_path, filename)
            text += extract_text_from_pdf(pdf_file)
    text = pdf_clean_text(text)
    with open(output_file, 'w') as file:
        file.write(text)

def clean_text(text):
    # add any text cleaning logic here
    return text

def consolidate_txt_files_to_txt(dir_path, output_file):
    text = ''
    for filename in os.listdir(dir_path):
        if filename.endswith('.txt'):
            txt_file = os.path.join(dir_path, filename)
            with open(txt_file, 'r') as file:
                text += file.read()
    text = clean_text(text)
    with open(output_file, 'w') as file:
        file.write(text)


# consolidate_pdf_text_to_txt('./data', 'consolidated_pdf_text.txt')
consolidate_pdf_text_to_txt('./data/new/', 'consolidated_text_ebooks.txt')
