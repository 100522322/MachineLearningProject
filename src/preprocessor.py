import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class Preprocessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        self.X = None
        self.y = None
    
    def load_data(self):
        """
        Loads the clean CSV data
        """
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"File not found: {self.filepath}")

        self.data = pd.read_csv(self.filepath)
        return self
    
    def preprocess_data(self):
        """
        Tranform text into numerical values
        """
        if self.data is None:
            raise ValueError("Data not loaded")
        
        # Split features and target
        target_column = 'price'
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]

        # Identify numerical and categorical columns
        numeric_cols = ['year', 'odometer']
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

        # Define the ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_cols),
                ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols)
            ],
            verbose_feature_names_out=False
        )

        self.X = preprocessor.fit_transform(X)
        self.y = y.to_numpy()

        return self
    
    def save_preprocessed_data(self, X_preprocessed_filepath, y_preprocessed_filepath):
        if self.X is None or self.y is None:
            raise ValueError("Data not preprocessed")

        np.save(X_preprocessed_filepath, self.X)
        np.save(y_preprocessed_filepath, self.y)

        return self
