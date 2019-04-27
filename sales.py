import argparse
from collections import namedtuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
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
        print(std_scaled_feats)
        training_X, testing_X, training_Y, testing_Y = train_test_split(
            features, labels, random_state=42
        )
    elif scaling == "MinMax":
        scaler = MinMaxScaler()
        minmax_feats = scaler.fit_transform(features)
        print(minmax_feats)
        training_X, testing_X, training_Y, testing_Y = train_test_split(
            minmax_feats, labels, random_state=42
        )
    else:
        training_X, testing_X, training_Y, testing_Y = train_test_split(
            features, labels, random_state=42
        )

    return training_X, testing_X, training_Y, testing_Y


def main():
    """
        Main execution point of our function
    """
    # Import df, create a separate one for our labels, and then drop
    # redundant columns
    sales_data = pd.read_csv("wholesale_customers_data.csv")
    channels = sales_data["Channel"]
    sales_data.drop(labels=["Channel", "Region"], axis=1, inplace=True)

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


if __name__ == "__main__":
    main()
