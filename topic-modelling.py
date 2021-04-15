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

lemmatizer = WordNetLemmatizer()

def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS:
            result.append(WordNetLemmatizer().lemmatize(token, 'v')) # for now, lemmatizing only verbs
    return result


reviews = ds["review_body"]
less_reviews = reviews[:1000]
processed = []
for review in less_reviews:
    processed.append(preprocess(review))

dictionary = gensim.corpora.Dictionary(processed) # construct word<->id mappings - it does it in alphabetical order

bow_corpus = [dictionary.doc2bow(review) for review in processed]

lda_model =  gensim.models.LdaModel(bow_corpus, 
                                   num_topics=20, 
                                   id2word=dictionary,                                    
                                   passes=10,
                                   ) 
lda_model.save('model2/lda.model')

rtime = time() - rtime
print("The program ran in", rtime, "seconds")