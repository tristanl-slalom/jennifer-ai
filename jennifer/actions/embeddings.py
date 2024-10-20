from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from openai import OpenAI
from pandas import DataFrame
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, TextColumn
from sklearn.cluster import KMeans

from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import numpy as np
from ast import literal_eval

from jennifer.actions.reviews.data import load_review_data, add_embedding_column, safely_write_to_csv
from jennifer.utilities.embeddings import (
    plot_multiclass_precision_recall,
    cosine_similarity,
    create_embedding,
)
from jennifer.utilities.plots import PLOT_COLORS


def embeddings_action(
    rebuild: bool,
    show_visualization: bool,
    show_classification: bool,
    show_clustering: bool,
    product_description: Optional[str],
    num_results: Optional[int],
    num_clusters: Optional[int],
    num_reviews_per_cluster: Optional[int],
    min_similarity: float,
):
    """
    Following the tutorial as laid out in our week-5 presentation. Use arguments to determine
    whether we want to show various matplot graphs. Gather the data and calculate embeddings
    with the latest models. Use an optional product description to show related reviews.
    Specify a number of clusters to group the reviews into a number of high level
    commonalities. You can specify both, but this may not work with every search term,
    number of clusters, and the amount of data we have. Expect errors!
    """
    client = OpenAI()

    dataset = _obtain_dataset(client, rebuild)

    if show_visualization:
        _visualize_in_2d(dataset)

    if show_classification:
        _classification_using_embeddings(dataset)

    if product_description and num_results:
        dataset = _text_search_using_embeddings(client, dataset, product_description, num_results)

    if num_clusters and num_reviews_per_cluster:
        _clustering_using_embeddings(
            client,
            dataset,
            show_clustering,
            num_clusters,
            num_reviews_per_cluster,
            min_similarity,
        )


def _obtain_dataset(client: OpenAI, rebuild: bool) -> DataFrame:
    """
    The filtered dataset without embeddings is added as a resource of this project.
    We'll calculate embeddings for the data and store that on our local machine for
    reuse.
    """
    max_tokens = 8000  # the maximum for text-embedding-3-small is 8191

    # load & inspect dataset
    input_directory = Path("jennifer") / "data"
    input_file_path = input_directory / "fine_food_reviews_1k.csv"
    output_directory = Path("output") / "embeddings"
    output_file_path = output_directory / f"{input_file_path.stem}_with_embeddings.csv"

    if output_file_path.exists() and not rebuild:
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn()) as progress:
            progress.add_task("Loading existing embeddings...")
            # Read the cached dataframe with the embeddings string, but not the numpy array we need.
            df = pd.read_csv(output_file_path)

            # Regenerate the numpy array before returning it. Here, having just been loaded from a
            # CSV, 'embeddings_string' is actually a string. Evaluating it will turn it back into
            # an object array (horribly unsafely, I might add), and then we can turn it into a numpy
            # array we need for calculations.
            df["embeddings"] = df.embeddings_string.apply(literal_eval).apply(np.array)
            return df

    # original source: https://github.com/openai/openai-cookbook/blob/main/examples/data/fine_food_reviews_1k.csv
    reviews_dataframe = load_review_data(input_file_path, max_tokens)

    # embedding column is all the embedding calculations, but there's no way to store a real array in a CSV
    # so this is it in a giant string that can be easily turned into a numpy array later.
    reviews_dataframe = add_embedding_column(client, reviews_dataframe)

    # Write the CSV without the numpy array.
    safely_write_to_csv(reviews_dataframe, output_file_path)

    # Add the embedding array into the in-memory dataframe, so it can be used throughout.
    # Because the embeddings_string column was just calculated, 'embeddings_string' is already
    # an object array, so all we need to do here is convert it to a numpy array.
    reviews_dataframe["embeddings"] = reviews_dataframe.embeddings_string.apply(np.array)

    return reviews_dataframe


def _visualize_in_2d(df: DataFrame):
    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
    ) as progress:
        progress.add_task("Reducing Dimensionality...")

        # Convert to a list of lists of floats
        matrix = np.array(df.embeddings.to_list())

        # Create a t-SNE model and transform the data
        tsne = TSNE(n_components=2, perplexity=15, random_state=42, init="random", learning_rate=200)
        vis_dims = tsne.fit_transform(matrix)
        print(vis_dims.shape)

    with Progress() as progress:
        task = progress.add_task("Showing Chart... (close to continue)", total=5)

        colors = ["red", "darkorange", "gold", "turquoise", "darkgreen"]
        x = [x for x, y in vis_dims]
        y = [y for x, y in vis_dims]
        color_indices = df.Score.values - 1

        colormap = matplotlib.colors.ListedColormap(colors)
        plt.scatter(x, y, c=color_indices, cmap=colormap, alpha=0.3)
        for score in [0, 1, 2, 3, 4]:
            progress.advance(task)

            avg_x = np.array(x)[df.Score - 1 == score].mean()
            avg_y = np.array(y)[df.Score - 1 == score].mean()
            color = colors[score]
            plt.scatter(avg_x, avg_y, marker="x", color=color, s=100)

        plt.title("Amazon ratings visualized in language using t-SNE")
        plt.show()


def _classification_using_embeddings(df: DataFrame):
    # split data into train and test
    x_train, x_test, y_train, y_test = train_test_split(
        list(df.embeddings.values), df.Score, test_size=0.2, random_state=42
    )

    # train random forest classifier
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)
    probabilities = clf.predict_proba(x_test)

    report = classification_report(y_test, predictions)

    print(report)

    plot_multiclass_precision_recall(probabilities, y_test, [1, 2, 3, 4, 5], clf)


def _text_search_using_embeddings(client: OpenAI, df: DataFrame, product_description: str, num_results: int):
    product_embedding = create_embedding(client, product_description)

    df["similarity"] = df.embeddings.apply(lambda x: cosine_similarity(x, product_embedding))

    results = (
        df.sort_values("similarity", ascending=False)
        .head(num_results)
        .combined.str.replace("Title: ", "")
        .str.replace("; Content:", ": ")
    )
    for r in results:
        print(r[:200])
        print()

    return df.sort_values("similarity", ascending=False)


def _clustering_using_embeddings(
    client: OpenAI,
    df: DataFrame,
    show_clustering: bool,
    num_clusters: int,
    num_reviews_per_cluster: int,
    min_similarity: float,
):
    if "similarity" in df:
        print("Similarities:")
        print(f"Max: {df['similarity'].max()}")
        print(f"Avg: {df['similarity'].mean()}")
        print(f"Num: {len(df[df.similarity > min_similarity])} results above {min_similarity}")
        df = df[df.similarity > min_similarity]

    matrix = np.vstack(df.embeddings.values)
    kmeans = KMeans(n_clusters=num_clusters, init="k-means++", n_init=10, random_state=42)
    kmeans.fit(matrix)
    labels = kmeans.labels_
    df["cluster"] = labels

    for i in range(num_clusters):
        print(f"Cluster {i+1} ({PLOT_COLORS[i]}) Theme:", end=" ")

        try:
            reviews = "\n".join(
                df[df.cluster == i]
                .combined.str.replace("Title: ", "")
                .str.replace("\n\nContent: ", ":  ")
                .sample(num_reviews_per_cluster, random_state=42)
                .values
            )

            messages = [
                {
                    "role": "user",
                    "content": f"What do the following customer reviews have in common?\n\n"
                    f'Customer reviews:\n"""\n{reviews}\n"""\n\nTheme:',
                }
            ]

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0,
                max_tokens=64,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            print(response.choices[0].message.content.replace("\n", ""))

            sample_cluster_rows = df[df.cluster == i].sample(num_reviews_per_cluster, random_state=42)
            for j in range(num_reviews_per_cluster):
                print(sample_cluster_rows.Score.values[j], end=", ")
                print(sample_cluster_rows.Summary.values[j], end=":   ")
                print(sample_cluster_rows.Text.str[:70].values[j])

            print("-" * 100)
        except ValueError:
            print(
                f"ERROR: Not enough values to cluster these results {num_clusters} times."
                f"Try broadening the search, reducing the clusters, or relaxing the score"
            )
            exit(1)

    if show_clustering:
        df.groupby("cluster").Score.mean().sort_values()

        tsne = TSNE(n_components=2, perplexity=15, random_state=42, init="random", learning_rate=200)
        vis_dims2 = tsne.fit_transform(matrix)

        x = [x for x, y in vis_dims2]
        y = [y for x, y in vis_dims2]

        if num_clusters > len(PLOT_COLORS):
            raise ValueError(f"Can't do more than {len(PLOT_COLORS)} clusters, sorry")

        for category, color in enumerate(PLOT_COLORS[:num_clusters]):
            xs = np.array(x)[df.cluster == category]
            ys = np.array(y)[df.cluster == category]
            plt.scatter(xs, ys, color=color, alpha=0.3)

            avg_x = xs.mean()
            avg_y = ys.mean()

            plt.scatter(avg_x, avg_y, marker="x", color=color, s=100)
        plt.title("Clusters identified visualized in language 2d using t-SNE")
        plt.show()
