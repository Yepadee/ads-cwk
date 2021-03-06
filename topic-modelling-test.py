import pandas as pd
from time import time

from nltk.stem import WordNetLemmatizer
from gensim.parsing.preprocessing import STOPWORDS
import gensim

DATASET = "amazon_reviews_us_Grocery_v1_00.tsv"

rtime = time()

print("loading dataset...")
ds = pd.read_table(DATASET, error_bad_lines=False, header=0, warn_bad_lines=False)
print("done!")

# Drop null rows
ds = ds.dropna()

lda = gensim.models.LdaModel.load('model/lda.model')
dct = gensim.corpora.Dictionary.load('model/lda.model.id2word')

lemmatizer = WordNetLemmatizer()

def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS:
            result.append(WordNetLemmatizer().lemmatize(token, 'v')) # for now, lemmatizing only verbs
    return result


reviews = ds["review_body"]
less_reviews = reviews[:100]
processed = []
for review in less_reviews:
    processed.append(preprocess(review))

#dictionary = gensim.corpora.Dictionary(processed) # construct word<->id mappings - it does it in alphabetical order

bow_corpus = [dct.doc2bow(review) for review in processed]
print(bow_corpus)
f=open("bow_corpus.txt","w+")
f.write(str(bow_corpus))
f.close()
print(dct)
for idx, topic in lda.print_topics(-1): # The words occuring in each class, and the weight given for that
    print("Topic: {} \nWords: {}".format(idx, topic ))
    print("\n")

rtime = time() - rtime
print("The program ran in", rtime, "seconds")