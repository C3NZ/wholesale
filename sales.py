import argparse

import pandas as pd


def main():
    # Import df
    sales_data = pd.read_csv("wholesale_customers_data.csv")

    # Slice out channels
    channels = sales_data["Channel"]

    sales_data.drop(labels=["Channel", "Region"], axis=1, inplace=True)
    print(sales_data)


if __name__ == "__main__":
    main()
