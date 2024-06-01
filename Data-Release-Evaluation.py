# data_release_evaluation.py

"""
Data Release Evaluation Module for Privacy-Preserving Data Release

This module contains functions for evaluating the utility and privacy guarantees of the differentially private data release.

Techniques Used:
- Utility Metrics
- Privacy Budget Analysis
- Re-identification Risk Assessment

Libraries/Tools:
- pandas
- numpy
- scikit-learn
- matplotlib

"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
import os

class DataReleaseEvaluation:
    def __init__(self):
        """
        Initialize the DataReleaseEvaluation class.
        """
        pass

    def load_data(self, original_data_filepath, private_data_filepath):
        """
        Load the original and differentially private data from CSV files.
        
        :param original_data_filepath: str, path to the original data CSV file
        :param private_data_filepath: str, path to the private data CSV file
        :return: tuple, original and private DataFrames
        """
        original_data = pd.read_csv(original_data_filepath)
        private_data = pd.read_csv(private_data_filepath)
        return original_data, private_data

    def evaluate_utility(self, original_data, private_data):
        """
        Evaluate the utility of the private data by comparing it to the original data.
        
        :param original_data: DataFrame, original data
        :param private_data: DataFrame, private data
        :return: dict, utility metrics
        """
        mse = mean_squared_error(original_data, private_data)
        utility_metrics = {
            'MSE': mse
        }
        return utility_metrics

    def evaluate_reidentification_risk(self, original_data, private_data):
        """
        Evaluate the re-identification risk of the private data.
        
        :param original_data: DataFrame, original data
        :param private_data: DataFrame, private data
        :return: float, re-identification risk score
        """
        matches = (original_data == private_data).sum().sum()
        total_entries = np.prod(original_data.shape)
        reidentification_risk = matches / total_entries
        return reidentification_risk

    def plot_comparison(self, original_data, private_data, output_dir):
        """
        Plot the comparison between original and private data.
        
        :param original_data: DataFrame, original data
        :param private_data: DataFrame, private data
        :param output_dir: str, directory to save the plot
        """
        os.makedirs(output_dir, exist_ok=True)

        for column in original_data.columns:
            plt.figure(figsize=(10, 6))
            plt.hist(original_data[column], bins=30, alpha=0.5, label='Original')
            plt.hist(private_data[column], bins=30, alpha=0.5, label='Private')
            plt.title(f'Comparison of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.legend(loc='upper right')
            plt.savefig(os.path.join(output_dir, f'comparison_{column}.png'))
            plt.show()

    def evaluate(self, original_data_filepath, private_data_filepath, output_dir):
        """
        Execute the full evaluation pipeline.
        
        :param original_data_filepath: str, path to the original data file
        :param private_data_filepath: str, path to the private data file
        :param output_dir: str, directory to save the evaluation results
        """
        # Load data
        original_data, private_data = self.load_data(original_data_filepath, private_data_filepath)

        # Evaluate utility
        utility_metrics = self.evaluate_utility(original_data, private_data)
        print("Utility Metrics:")
        for metric, value in utility_metrics.items():
            print(f"{metric}: {value}")

        # Evaluate re-identification risk
        reidentification_risk = self.evaluate_reidentification_risk(original_data, private_data)
        print(f"Re-identification Risk: {reidentification_risk}")

        # Plot comparison
        self.plot_comparison(original_data, private_data, output_dir)

if __name__ == "__main__":
    original_data_filepath = 'data/processed/processed_data.csv'
    private_data_filepath = 'data/private/laplace_data.csv'  # Example private data file
    output_dir = 'results/evaluation/'
    os.makedirs(output_dir, exist_ok=True)

    evaluator = DataReleaseEvaluation()

    # Evaluate the data release
    evaluator.evaluate(original_data_filepath, private_data_filepath, output_dir)
    print("Data release evaluation completed and results saved.")
