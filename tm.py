import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

DATASET = "amazon_reviews_us_Grocery_v1_00.tsv"

print("loading dataset...")
ds = pd.read_table(DATASET, error_bad_lines=False, header=0, warn_bad_lines=False)
print("done!")

ds = ds.dropna()

sia = SentimentIntensityAnalyzer()

def sentiment_heuristic(row):
    return sia.polarity_scores(row)['compound']

#ds = ds.head()

print("calculating sentiment...")
ds["sentiment_score"] = ds["review_body"].apply(sentiment_heuristic)
print("done!")

ds.to_csv("sentiment_ds.csv")

print(ds)