# latent_factor_models.py

"""
Latent Factor Models Module for Privacy-Preserving Data Release

This module contains functions for developing and training latent factor models to represent the underlying structure of mixed-type data.

Techniques Used:
- Matrix Factorization
- Bayesian Latent Factor Models

Libraries/Tools:
- pandas
- numpy
- scikit-learn
- TensorFlow or PyTorch

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

class LatentFactorModel:
    def __init__(self, num_factors=10, epochs=50, batch_size=32):
        """
        Initialize the LatentFactorModel class.
        
        :param num_factors: int, number of latent factors
        :param epochs: int, number of training epochs
        :param batch_size: int, size of the training batches
        """
        self.num_factors = num_factors
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None

    def build_model(self, num_features):
        """
        Build the latent factor model using Keras.
        
        :param num_features: int, number of input features
        :return: compiled Keras model
        """
        inputs = keras.Input(shape=(num_features,))
        x = layers.Dense(self.num_factors, activation='relu')(inputs)
        x = layers.Dense(self.num_factors, activation='relu')(x)
        outputs = layers.Dense(num_features, activation='sigmoid')(x)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='mse')
        return model

    def load_data(self, filepath):
        """
        Load data from a CSV file.
        
        :param filepath: str, path to the CSV file
        :return: DataFrame, loaded data
        """
        data = pd.read_csv(filepath)
        return data

    def preprocess_data(self, data):
        """
        Preprocess the data by splitting into training and testing sets.
        
        :param data: DataFrame, input data
        :return: tuple, training and testing data
        """
        X_train, X_test = train_test_split(data, test_size=0.2, random_state=42)
        return X_train, X_test

    def train_model(self, X_train):
        """
        Train the latent factor model.
        
        :param X_train: DataFrame, training features
        """
        self.model.fit(X_train, X_train, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.2)

    def save_model(self, model_dir, model_name='latent_factor_model.h5'):
        """
        Save the trained model to a file.
        
        :param model_dir: str, directory to save the model
        :param model_name: str, name of the model file
        """
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, model_name)
        self.model.save(model_path)
        print(f"Model saved to {model_path}")

    def load_trained_model(self, model_dir, model_name='latent_factor_model.h5'):
        """
        Load a trained model from a file.
        
        :param model_dir: str, directory containing the trained model
        :param model_name: str, name of the model file
        :return: loaded Keras model
        """
        model_path = os.path.join(model_dir, model_name)
        model = keras.models.load_model(model_path)
        return model

if __name__ == "__main__":
    data_filepath = 'data/processed/processed_data.csv'
    model_dir = 'models/'
    num_factors = 10
    epochs = 50
    batch_size = 32

    lfm = LatentFactorModel(num_factors, epochs, batch_size)

    # Load and preprocess data
    data = lfm.load_data(data_filepath)
    X_train, X_test = lfm.preprocess_data(data)

    # Build model
    num_features = X_train.shape[1]
    lfm.model = lfm.build_model(num_features)

    # Train model
    lfm.train_model(X_train)
    print("Model training completed.")

    # Save model
    lfm.save_model(model_dir)
