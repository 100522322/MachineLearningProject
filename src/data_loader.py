import pandas as pd
import os
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
    
    def load_data(self, nrows=None):
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"File not found: {self.filepath}")

        self.data = pd.read_csv(self.filepath, nrows=nrows)
        return self.data
    
    def clean_data(self):
        if self.data is None:
            raise ValueError("Data not loaded")
        
        drop_cols = ['id', 'url', 'region_url', 'image_url', 'description', 'lat', 'long', 'VIN', 'region', 'model']

        self.data = self.data.drop(columns=drop_cols, errors='ignore')

        # Drop rows with missing price
        self.data = self.data.dropna(subset=['price'])

        # Filter outliers
        self.data = self.data[(self.data['price'] > 500) and (self.data['price'] < 100000)]
        self.data = self.data((self.data['year'] > 1980))
        self.data = self.data[(self.data['odometer'] < 400000)]

        return self.data
    
    def save_clean_data(self, output_filepath):
        if self.data is None:
            raise ValueError("Data not loaded or cleaned")
        
        self.data.to_csv(output_filepath, index=False)
    
    def get_data_split(self, target='price', test_size=0.2):
        X = self.data.drop(columns=[target])
        y = self.data[target]

        return train_test_split(X, y, test_size=test_size)