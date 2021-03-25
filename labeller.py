import pandas as pd
from rich.console import Console
import click


DATASET = "" # path to the negative dataset csv 
OUTPUT_FOLDER = "labelled_dataset"


class Labeller:
    def __init__(self, dataset_path, start, end, copy=True):
        self.start = start
        self.end = end
        self.size = start-end
        self.dataset_path = dataset_path
        self.dataset = self._parse_dataset(copy)

    def save(self, output_folder):
        name = f"{self.start}-{self.end}_reviews.csv"
        with open(f"{output_folder}/{name}", "w") as text_file:
            text_file.write(self.dataset.to_csv(index=False))

    def write_labels_interactively(self, feature_column_name, label_column_name):
        labels = []
        console = Console()
        for (_, row) in self.dataset.iterrows():
            feature = row[feature_column_name]
            label = console.input(f"[bold red]{feature}?")
            console.print(f"[bold green]--------------------------------")
            labels.append(label)

        self.dataset[label_column_name] = labels

    def _parse_dataset(self, copy=True):
        chunk = pd.read_table(self.dataset_path,  error_bad_lines=False,
                              skiprows=range(1, self.start), nrows=self.end-self.start+1, header=0)
        return chunk.copy() if copy else chunk


@click.command()
@click.option('--start', help='Start row index .', type=int)
@click.option('--end', help='End row index.', type=int)
def main(start, end):
    """
    A script that allows interactive labelling of a dataset. It saves a copy of the newly labelled dataset into the labelled_dataset folder  
    """
    labeller = Labeller(DATASET, start, end)
    labeller.write_labels_interactively("review_body", "food_safety_flag")
    labeller.save(OUTPUT_FOLDER)


if __name__ == "__main__":
    main()
