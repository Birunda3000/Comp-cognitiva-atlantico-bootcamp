<<<<<<< HEAD:atividades_classe/Aula_2 - 14-05-2022/code1.py
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert (1, '../src')

import os
from nlp_utils import get_sample_article

# Import Counter and word_tokenize
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
from collections import Counter
=======
# Import Counter and word_tokenize
import tensorflow

__
__

import tensorflow
from NLP.src.nlp_utils import get_sample_article
>>>>>>> 45b50816698b6c1ee1584632825f277c990ad09a:atividades_classe/NLP/Aula_2 - 14-05-2022/code1.py

article = get_sample_article()

# Tokenize the article: tokens
tokens = word_tokenize(article)

# Convert the tokens into lowercase: lower_tokens
lower_tokens = [t.lower() for t in tokens]

# Create a Counter with the lowercase tokens: bow_simple
bow_simple = Counter(lower_tokens)

# Print the 10 most common tokens
print(bow_simple.most_common(10))