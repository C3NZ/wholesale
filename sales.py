import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


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
    # Import df
    sales_data = pd.read_csv("wholesale_customers_data.csv")

    # Slice out channels
    channels = sales_data["Channel"]

    sales_data.drop(labels=["Channel", "Region"], axis=1, inplace=True)
    print(sales_data.to_numpy())
    visualize_correlation(sales_data)


if __name__ == "__main__":
    main()
