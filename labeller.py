import pandas as pd
from rich.console import Console
import click


DATASET = "bad_reviews_refined.csv" # path to the negative dataset csv 
OUTPUT_FOLDER = "labelled_dataset"


class Labeller:
    def __init__(self, dataset_path, start, end, copy=True):
        self.start = start
        self.end = end
        self.size = start-end
        self.dataset_path = dataset_path
        self.dataset = pd.read_csv(DATASET)
        self.dataset = self._parse_dataset()

    def save(self, output_folder):
        name = f"{self.start}-{self.end}_reviews.csv"
        with open(f"{output_folder}/{name}", "w") as text_file:
            text_file.write(self.dataset.to_csv(index=False))

    def write_labels_interactively(self, feature_column_names, label_column_name):
        labels = []
        console = Console()
        for i, row in self.dataset.iterrows():
            console.print(f"[bold green]--------------------------------")
            for feature_name in feature_column_names:
                feature = row[feature_name]
                console.print(f"[bold yellow] {feature_name} [bold red]{feature}?")

            console.print(f"[bold green]--------------------------------")
            label = console.input(f"[bold blue] Food safety issue (1) or not (0)?")
            self.dataset.at[i, label_column_name] = label
        

    def _parse_dataset(self):
        df = pd.read_csv(DATASET)
        sliced_df = pd.DataFrame(columns=df.columns)
        for i in range(self.start, self.end):
            sliced_df.loc[i] = df.loc[i]
        return sliced_df



@click.command()
@click.option('--start', help='Start row index .', type=int)
@click.option('--end', help='End row index.', type=int)
def main(start, end):
    """
    A script that allows interactive labelling of a dataset. It saves a copy of the newly labelled dataset into the labelled_dataset folder  
    """
    labeller = Labeller(DATASET, start, end)
    labeller.write_labels_interactively(["review_headline", "review_body", "sentiment_score"], "food_safety_flag")
    labeller.save(OUTPUT_FOLDER)


if __name__ == "__main__":
    main()
