import os
import pathlib
from pathlib import Path
import re

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

def txt_check (string):
    if '.txt' in string:
        return True
    else:
        return False

def clean(string):
    
    #string = re.sub(u'(0-9)', '', string)
    #string = re.sub(u'[^a-zA-Z0-9áéíóúÁÉÍÓÚâêîôÂÊÎÔãõÃÕçÇ: ]', '', string)
    string = re.sub(u'[^a-zA-ZáéíóúÁÉÍÓÚâêîôÂÊÎÔãõÃÕçÇ: ]', '', string)

    string = string.lower()

    string_list = re.split(r"\s+|-", string)
    
    return [word for word in string_list if not word in stopwords.words('portuguese')]

DATADIR = "text"
#DATADIR = str(input("Caminho dos documentos"))
DATADIR = pathlib.Path(DATADIR)

OUTDIR = "cleaned-no-lem"
OUTDIR = pathlib.Path(OUTDIR)

texts = os.listdir(DATADIR)
texts.sort()

filtered_object = filter(txt_check, texts)
texts = list(filtered_object)

for text in texts:
    
    path = os.path.join(DATADIR, text)
    
    temp =  Path(path).read_text().replace('\n', ' ')
    
    temp = clean(temp)
    
    with open('cleaned-no-lem/' + 'cleaned-' + text, "w") as text_file:
        for word in temp:
            text_file.write(word+'\n')