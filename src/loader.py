"""
This module provides a class for loading news related datasets from a given path.
It uses pandas for data loading and argparse for command line argument parsing.
"""

import argparse
import pandas as pd


class NewsDataLoader:  # pylint: disable=too-few-public-methods
    """
    A class that loads news related datasets when provided a path.
    """

    def __init__(self):
        """
        Initializes the NewsDataLoader with an empty dictionary to store loaded data.
        """
        self.data = {}

    def load_data(self, path):
        """
        Loads data from a CSV file at the given path.

        Parameters:
        path (str): The path to the CSV file.

        Returns:
        DataFrame: The loaded data.
        """
        if path not in self.data:
            self.data[path] = pd.read_csv(path)
        return self.data[path]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export news history')
    parser.add_argument('--zip', help="Name of a zip file to import")
    args = parser.parse_args()
