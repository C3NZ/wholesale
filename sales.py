import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def register_parser():
    parser = argparse.ArgumentParser(
        description="Applying machine learning to wholesale sales"
    )

    parser.add_argument(
        "-c",
        "--correlate",
        dest="correlate",
        action="store_true",
        help="Show the correlation between all of the features in our dataset",
    )

    parser.add_argument(
        "-s",
        "--scale",
        dest="scale",
        action="store_true",
        help="Scale our data using both minmax and standard scaling",
    )

    parser.add_argument(
        "-m",
        "--minmax",
        des="minmax",
        action="store_true",
        help="Scaleour data using minmax scaling",
    )

    return parser


def create_model():
    pass


def visualize_correlation(sales_data: pd.DataFrame):
    """
        Visualize the correlation between all variables
    """
    correlation = sales_data.corr()
    sns.heatmap(
        correlation, xticklabels=correlation.columns, yticklabels=correlation.columns
    )
    plt.show()


def main():
    # Import df, create a separate one for our labels, and then drop
    # redundant columns
    sales_data = pd.read_csv("wholesale_customers_data.csv")
    channels = sales_data["Channel"]
    sales_data.drop(labels=["Channel", "Region"], axis=1, inplace=True)

    parser = register_parser()

    args = parser.parse_args()

    if args.correlate:
        visualize_correlation(sales_data)


if __name__ == "__main__":
    main()
