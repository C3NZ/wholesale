import argparse
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def register_parser():
    """
        Register the argument parser and all of the command arguments
    """
    parser = argparse.ArgumentParser(
        description="Applying machine learning to wholesale sales"
    )

    parser.add_argument(
        "-c",
        "--correlate",
        dest="correlate",
        action="store_true",
        help="Show the correlation between all of the features in our dataset",
        default=False,
    )

    parser.add_argument(
        "-a",
        "--all",
        dest="all",
        action="store_true",
        help="Scale our data using both minmax and standard scaling",
        default=False,
    )

    parser.add_argument(
        "-m",
        "--minmax",
        dest="minmax",
        action="store_true",
        help="Scale our data using minmax scaling",
        default=False,
    )

    parser.add_argument(
        "-s",
        "--standard",
        dest="standard",
        action="store_true",
        help="Scale our data using standard scaling",
        default=False,
    )

    return parser


def visualize_correlation(sales_data: pd.DataFrame):
    """
        Visualize the correlation between all variables
    """
    correlation = sales_data.corr()
    sns.heatmap(
        correlation, xticklabels=correlation.columns, yticklabels=correlation.columns
    )
    plt.show()


def get_model_data(features: pd.DataFrame, labels: pd.DataFrame, scaling: str = "None"):
    """
        Obtain the training/test data split
    """
    # Initial assignment of model data
    training_X = testing_X = training_Y = testing_Y = None

    if scaling == "Standard":
        scaler = StandardScaler()
        std_scaled_feats = scaler.fit_transform(features)
        training_X, testing_X, training_Y, testing_Y = train_test_split(
            std_scaled_feats, labels, random_state=42
        )
    elif scaling == "MinMax":
        scaler = MinMaxScaler()
        minmax_feats = scaler.fit_transform(features)
        training_X, testing_X, training_Y, testing_Y = train_test_split(
            minmax_feats, labels, random_state=42
        )
    else:
        training_X, testing_X, training_Y, testing_Y = train_test_split(
            features, labels, random_state=42
        )

    return training_X, testing_X, training_Y, testing_Y


def calculate_pca(model_data: tuple, n_components: int = 2):
    """
        Reduce the dimensionality of given model data using PCA.
    """
    data_scaling = model_data[0]
    training_X, testing_X, training_Y, testing_Y = model_data[1]

    pca = PCA(n_components=n_components)
    print("--- START PCA ---")
    print(
        f"Reducing dimensionality of our data that has {data_scaling} to {n_components} components\n"
    )
    reduced_dimensions = pca.fit_transform(training_X)

    info_preserved = pca.explained_variance_ratio_

    print("Information preserved through each component:")
    print(info_preserved)

    print(f"Total information preserved: {sum(info_preserved)}")
    print("--- END PCA ---\n")
    return reduced_dimensions


def create_cluster(data: np.ndarray):
    """
        Create the clusters
    """
    cluster = KMeans(n_clusters=2)
    cluster.fit_transform(data)
    return cluster


def visualize_reduced_cluster(points: np.ndarray, clusters: np.ndarray):
    """
        Visualize our dimensionally reduced data and it's' clusters
    """
    print("Drawing")
    sns.scatterplot(x=points[:, 0], y=points[:, 1])
    sns.scatterplot(x=clusters[:, 0], y=clusters[:, 1])
    plt.show()


def main():
    """
        Main execution point of our function
    """
    # Import df, create a separate one for our labels, and then
    # redundant columns
    sales_data = pd.read_csv("wholesale_customers_data.csv")
    channels = sales_data["Channel"]
    sales_data.drop(labels=["Channel", "Region"], axis=1, inplace=True)

    # Collection of all of our data
    all_model_data = []
    all_model_data.append(("No scaling", get_model_data(sales_data, channels)))

    parser = register_parser()

    args = parser.parse_args()

    if args.correlate:
        visualize_correlation(sales_data)

    if args.standard:
        all_model_data.append(
            (
                "Standard scaling",
                get_model_data(sales_data, channels, scaling="Standard"),
            )
        )

    if args.minmax:
        all_model_data.append(
            ("MinMax scaling", get_model_data(sales_data, channels, scaling="MinMax"))
        )

    for model_data in all_model_data:

        training_X, testing_X, training_Y, testing_Y = model_data[1]
        scaled_dimensions = calculate_pca(model_data)
        cluster = create_cluster(training_X)
        scaled_cluster = create_cluster(scaled_dimensions)
        visualize_reduced_cluster(scaled_dimensions, scaled_cluster.cluster_centers_)


if __name__ == "__main__":
    main()
