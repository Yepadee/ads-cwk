import pandas as pd
from nltk.stem import WordNetLemmatizer
import gensim
from time import time
import sys

DATASET = "amazon_reviews_us_Grocery_v1_00.tsv"

n_topics = sys.argv[1]

rtime = time()

print("loading dataset...")
ds = pd.read_table(DATASET, error_bad_lines=False, header=0, warn_bad_lines=False)
# Drop null rows
ds = ds.dropna()
print("done!")
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS:
            result.append(WordNetLemmatizer().lemmatize(token, 'v')) # for now, lemmatizing only verbs
    return result


reviews = ds["review_body"]
less_reviews = reviews
processed = []
for review in less_reviews:
    processed.append(preprocess(review))

dct = gensim.corpora.Dictionary.load(f'model{n_topics}/lda.model.id2word')

bow_corpus = [dct.doc2bow(review) for review in processed]

f=open(f"model{n_topics}/bow_corpus.txt","w+")
f.write(str(bow_corpus))
f.close()

rtime = time() - rtime
print("The program ran in", rtime, "seconds")