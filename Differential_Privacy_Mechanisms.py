# differential_privacy_mechanisms.py

"""
Differential Privacy Mechanisms Module for Privacy-Preserving Data Release

This module contains functions for implementing differentially private mechanisms to ensure the privacy of the data release.

Techniques Used:
- Laplace Mechanism
- Exponential Mechanism
- Randomized Response

Libraries/Tools:
- pandas
- numpy
- diffprivlib

"""

import pandas as pd
import numpy as np
from diffprivlib.mechanisms import Laplace, Exponential, RandomizedResponse
import os

class DifferentialPrivacy:
    def __init__(self, epsilon=1.0):
        """
        Initialize the DifferentialPrivacy class.
        
        :param epsilon: float, privacy budget parameter
        """
        self.epsilon = epsilon

    def load_data(self, filepath):
        """
        Load data from a CSV file.
        
        :param filepath: str, path to the CSV file
        :return: DataFrame, loaded data
        """
        data = pd.read_csv(filepath)
        return data

    def apply_laplace_mechanism(self, data, sensitivity):
        """
        Apply the Laplace mechanism to the data.
        
        :param data: DataFrame, input data
        :param sensitivity: float, sensitivity of the query
        :return: DataFrame, data with added Laplace noise
        """
        laplace_mech = Laplace(epsilon=self.epsilon, sensitivity=sensitivity)
        noisy_data = data.applymap(lambda x: laplace_mech.randomise(x))
        return noisy_data

    def apply_exponential_mechanism(self, data, utility_function, sensitivity):
        """
        Apply the Exponential mechanism to the data.
        
        :param data: DataFrame, input data
        :param utility_function: function, utility function for the exponential mechanism
        :param sensitivity: float, sensitivity of the utility function
        :return: DataFrame, data with selected values by exponential mechanism
        """
        exponential_mech = Exponential(epsilon=self.epsilon, sensitivity=sensitivity)
        selected_data = data.applymap(lambda x: exponential_mech.randomise(x, utility_function))
        return selected_data

    def apply_randomized_response(self, data, p, q):
        """
        Apply the Randomized Response mechanism to the data.
        
        :param data: DataFrame, input data
        :param p: float, probability of returning the true value
        :param q: float, probability of returning a random value
        :return: DataFrame, data with applied randomized response
        """
        rr_mech = RandomizedResponse(epsilon=self.epsilon, p=p, q=q)
        rr_data = data.applymap(lambda x: rr_mech.randomise(x))
        return rr_data

    def save_data(self, data, output_dir, filename):
        """
        Save the differentially private data to a file.
        
        :param data: DataFrame, data to save
        :param output_dir: str, directory to save the data
        :param filename: str, name of the output file
        """
        os.makedirs(output_dir, exist_ok=True)
        output_filepath = os.path.join(output_dir, filename)
        data.to_csv(output_filepath, index=False)
        print(f"Differentially private data saved to {output_filepath}")

if __name__ == "__main__":
    data_filepath = 'data/processed/processed_data.csv'
    output_dir = 'data/private/'
    epsilon = 1.0
    sensitivity = 1.0  # Example sensitivity value

    dp = DifferentialPrivacy(epsilon)

    # Load data
    data = dp.load_data(data_filepath)

    # Apply Laplace mechanism
    laplace_data = dp.apply_laplace_mechanism(data, sensitivity)
    dp.save_data(laplace_data, output_dir, 'laplace_data.csv')

    # Define a utility function for the exponential mechanism
    def utility_function(x):
        return -abs(x)

    # Apply Exponential mechanism
    exponential_data = dp.apply_exponential_mechanism(data, utility_function, sensitivity)
    dp.save_data(exponential_data, output_dir, 'exponential_data.csv')

    # Apply Randomized Response mechanism
    p = 0.8  # Example probability of returning the true value
    q = 0.2  # Example probability of returning a random value
    rr_data = dp.apply_randomized_response(data, p, q)
    dp.save_data(rr_data, output_dir, 'rr_data.csv')
