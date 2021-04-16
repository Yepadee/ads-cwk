import pandas as pd
from time import time

from nltk.stem import WordNetLemmatizer
from gensim.parsing.preprocessing import STOPWORDS
import gensim

DATASET = "amazon_reviews_us_Grocery_v1_00.tsv"

rtime = time()

# print("loading dataset...")
# ds = pd.read_table(DATASET, error_bad_lines=False, header=0, warn_bad_lines=False)
# print("done!")

lda = gensim.models.LdaModel.load('model/lda.model')
dct = gensim.corpora.Dictionary.load('model/lda.model.id2word')

print(dct)
for idx, topic in lda.print_topics(-1): # The words occuring in each class, and the weight given for that
    print("Topic: {} \nWords: {}".format(idx, topic ))
    print("\n")
rtime = time() - rtime
print("The program ran in", rtime, "seconds")