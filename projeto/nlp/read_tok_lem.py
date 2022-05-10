import os
import pathlib
import pdfplumber
import re
import nltk
nltk.download('stopwords')
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import stanza

def file_check (file_name, label='texto'):# Returns True or False
    return label in file_name

def read_files(DATADIR):
    DATADIR = pathlib.Path(DATADIR)
    PDFs = os.listdir(DATADIR)
    PDFs.sort()
    PDFs = filter(file_check, PDFs)
    PDFs = list(PDFs)
    texts = []
    for pdf in PDFs:
        path = os.path.join(DATADIR, pdf)
        temp = pdfplumber.open(path)
        t = ''
        for page in temp.pages:
            page = page.extract_text()        
            t=t+' '+page
        texts.append(t)
    return texts

def clean_special_characters(texts):
    cleaned_texts_list = []
    for text in texts:        
        text = re.sub(u'-', ' ', text)#palavras com -
        cleaned_text = re.sub(u'[^a-zA-ZáéíóúÁÉÍÓÚâêîôÂÊÎÔãõÃÕçÇñÑ ]', '', text)
        cleaned_text = cleaned_text.lower()
        cleaned_texts_list.append(cleaned_text)
    return cleaned_texts_list

def remove_stopwords(texts):
    texts_list = []    
    for text in texts:       
        text = text.split()       
        stopwords = nltk.corpus.stopwords.words('portuguese')
        stop = set(stopwords)      
        text_no_stop = [w.strip() for w in text]       
        text_no_stop = [w for w in text if w not in stop]       
        text_no_stop = " ".join(text_no_stop)        
        texts_list.append(text_no_stop)    
    return texts_list

def tokenize_lemantize(pathModelStanza, texts):
    nlp = stanza.Pipeline(lang='pt', processors='tokenize,lemma', model_dir=pathModelStanza)
    text_list = []
    for text in texts:    
        doc = nlp(text)
        text_list.append(doc)
    return text_list

def imprimir_lemantizacao(doc):
    print(*[f'word: {word.text+" "}\tlemma: {word.lemma}\n' for sent in doc.sentences for word in sent.words])

def write_text(texts, path=''):  
    for i in range(len(texts)):
        path_w = os.path.join(path,'cleaned_text-'+str(i)+'.txt')
        print(path_w)#exibe os arquivos criados
        with open(path_w, "w") as text_file:
            [text_file.write(f'{word.lemma}\n') for sent in texts[i].sentences for word in sent.words]

def pre_processing(path='', output_path='text', model_path='stanza_models'):
    texts = read_files(path)
    cleaned_text = clean_special_characters(texts)
    text_no_stop = remove_stopwords(cleaned_text)
    text_lemma = tokenize_lemantize(model_path, text_no_stop)
    write_text(texts=text_lemma, path=output_path)

pre_processing(path='docs', output_path='texts', model_path='stanza_models')