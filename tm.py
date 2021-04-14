import pandas as pd 
DATASET = "amazon_reviews_us_Grocery_v1_00.tsv"

ds = pd.read_table(DATASET, error_bad_lines=False, header=0, warn_bad_lines=False) 
ds = ds.dropna()
