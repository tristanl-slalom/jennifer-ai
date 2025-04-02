from typing import List

from matplotlib.lines import Line2D
from openai import AzureOpenAI
from scipy import spatial
from sklearn.metrics import average_precision_score, precision_recall_curve

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def distances_from_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    distance_metric="cosine",
) -> List[List]:
    """
    Originally this was a method in Open AI's utilities module, but they seemed
    to have removed it. I found this online; it seems to do the trick.

    So far we only seem to use cosine, and this function could be simplified to
    just do that, but I keep the original version I found here just in case.
    Basically, seems to take a list of floats and a bunch of lists of floats and
    calculates the cosine of each entry between them.

    Math stuff, I dunno. Fancy!
    """
    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }
    distances = [distance_metrics[distance_metric](query_embedding, embedding) for embedding in embeddings]
    return distances


def plot_multiclass_precision_recall(y_score, y_true_untransformed, class_list, classifier_name):
    """
    Precision-Recall plotting for a multiclass problem. It plots average precision-recall,
    per class precision recall and reference f1 contours.

    Code slightly modified, but heavily based on
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
    """
    n_classes = len(class_list)
    y_true = pd.concat([(y_true_untransformed == class_list[i]) for i in range(n_classes)], axis=1).values

    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_true[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision_micro, recall_micro, _ = precision_recall_curve(y_true.ravel(), y_score.ravel())
    average_precision_micro = average_precision_score(y_true, y_score, average="micro")
    print(
        str(classifier_name) + " - Average precision score over all classes: {0:0.2f}".format(average_precision_micro)
    )

    # setup plot details
    plt.figure(figsize=(9, 10))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    l: list[Line2D] = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append("iso-f1 curves")
    (l,) = plt.plot(recall_micro, precision_micro, color="gold", lw=2)
    lines.append(l)
    labels.append("average Precision-recall (auprc = {0:0.2f})" "".format(average_precision_micro))

    for i in range(n_classes):
        (l,) = plt.plot(recall[i], precision[i], lw=2)
        lines.append(l)
        labels.append(
            "Precision-recall for class `{0}` (auprc = {1:0.2f})" "".format(class_list[i], average_precision[i])
        )

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{classifier_name}: Precision-Recall curve for each class")
    plt.legend(lines, labels)
    plt.show()


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def create_embedding(client: AzureOpenAI, input_text: str):
    return client.embeddings.create(input=input_text, model="text-embedding-3-small").data[0].embedding
