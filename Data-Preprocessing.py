# data_preprocessing.py

"""
Data Preprocessing Module for Privacy-Preserving Data Release

This module contains functions for collecting, cleaning, normalizing, and preparing data for model training and evaluation.

Techniques Used:
- Data cleaning
- Normalization
- Feature extraction
- Handling missing data

Libraries/Tools:
- pandas
- numpy
- scikit-learn

"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import os

class DataPreprocessing:
    def __init__(self):
        """
        Initialize the DataPreprocessing class.
        """
        self.numeric_imputer = SimpleImputer(strategy='mean')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

    def load_data(self, filepath):
        """
        Load data from a CSV file.
        
        :param filepath: str, path to the CSV file
        :return: DataFrame, loaded data
        """
        data = pd.read_csv(filepath)
        return data

    def clean_data(self, data):
        """
        Clean the data by removing duplicates.
        
        :param data: DataFrame, input data
        :return: DataFrame, cleaned data
        """
        data = data.drop_duplicates()
        return data

    def handle_missing_data(self, data, numeric_features, categorical_features):
        """
        Handle missing data by imputing values.
        
        :param data: DataFrame, input data
        :param numeric_features: list, list of numeric feature names
        :param categorical_features: list, list of categorical feature names
        :return: DataFrame, data with imputed values
        """
        data[numeric_features] = self.numeric_imputer.fit_transform(data[numeric_features])
        data[categorical_features] = self.categorical_imputer.fit_transform(data[categorical_features])
        return data

    def normalize_data(self, data, numeric_features):
        """
        Normalize the numeric data using standard scaling.
        
        :param data: DataFrame, input data
        :param numeric_features: list, list of numeric feature names
        :return: DataFrame, normalized data
        """
        data[numeric_features] = self.scaler.fit_transform(data[numeric_features])
        return data

    def encode_categorical_data(self, data, categorical_features):
        """
        Encode categorical data using one-hot encoding.
        
        :param data: DataFrame, input data
        :param categorical_features: list, list of categorical feature names
        :return: DataFrame, data with encoded categorical features
        """
        encoded_features = self.encoder.fit_transform(data[categorical_features])
        encoded_df = pd.DataFrame(encoded_features, columns=self.encoder.get_feature_names_out(categorical_features))
        data = data.drop(columns=categorical_features)
        data = pd.concat([data, encoded_df], axis=1)
        return data

    def preprocess(self, raw_data_filepath, processed_data_dir, numeric_features, categorical_features):
        """
        Execute the full preprocessing pipeline.
        
        :param raw_data_filepath: str, path to the input data file
        :param processed_data_dir: str, directory to save processed data
        :param numeric_features: list, list of numeric feature names
        :param categorical_features: list, list of categorical feature names
        :return: DataFrame, preprocessed data
        """
        # Load data
        data = self.load_data(raw_data_filepath)

        # Clean data
        data = self.clean_data(data)

        # Handle missing data
        data = self.handle_missing_data(data, numeric_features, categorical_features)

        # Normalize numeric data
        data = self.normalize_data(data, numeric_features)

        # Encode categorical data
        data = self.encode_categorical_data(data, categorical_features)

        # Save processed data
        os.makedirs(processed_data_dir, exist_ok=True)
        processed_data_filepath = os.path.join(processed_data_dir, 'processed_data.csv')
        data.to_csv(processed_data_filepath, index=False)
        print(f"Processed data saved to {processed_data_filepath}")

        return data

if __name__ == "__main__":
    raw_data_filepath = 'data/raw/data.csv'
    processed_data_dir = 'data/processed/'
    numeric_features = ['feature1', 'feature2', 'feature3']  # Example numeric feature names
    categorical_features = ['feature4', 'feature5']  # Example categorical feature names

    preprocessing = DataPreprocessing()

    # Preprocess the data
    processed_data = preprocessing.preprocess(raw_data_filepath, processed_data_dir, numeric_features, categorical_features)
    print("Data preprocessing completed and data saved.")
