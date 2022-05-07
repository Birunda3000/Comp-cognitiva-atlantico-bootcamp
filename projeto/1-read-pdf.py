import os
import pathlib
import pdfplumber
import re

def pdf_check (string):
    # Returns True or False
    return '.pdf' in string

DATADIR = "docs"
DATADIR = str(input("Caminho dos documentos"))
DATADIR = pathlib.Path(DATADIR)

OUTDIR = "text"
OUTDIR = pathlib.Path(OUTDIR)

PDFs = os.listdir(DATADIR)
PDFs.sort()

filtered_object = filter(pdf_check, PDFs)
PDFs = list(filtered_object)

for pdf in PDFs:
    path = os.path.join(DATADIR, pdf)
    temp = pdfplumber.open(path)
    for page in temp.pages:
        with open('text/' + re.sub(r"[.]","-",pdf) + '.txt', "w") as text_file:
            text_file.write(page.extract_text())