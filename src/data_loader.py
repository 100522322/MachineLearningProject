import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
    
    def load_data(self, nrows=None):
        """Loads the data from the filepath inserted on init"""
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"File not found: {self.filepath}")

        print("Loading data...")
        chunks = []
        for chunk in tqdm(pd.read_csv(self.filepath, chunksize=10000, nrows=nrows), desc="Reading CSV"):
            chunks.append(chunk)
        self.data = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        print(f"Loaded {len(self.data)} rows")
        return self
    
    def clean_data(self):
        """Eliminates the columns that are not needed"""
        if self.data is None:
            raise ValueError("Data not loaded")

        print("\nCleaning data...")
        # TODO select what rows we have to drop
        drop_cols = ['id', 'url', 'region_url', 'image_url', 'description', 'lat', 'long',
                      'VIN', 'region', 'model', 'posting_date']

        tqdm.write("Dropping unnecessary columns...")
        self.data = self.data.drop(columns=drop_cols, errors='ignore')

        # Drop rows with missing price
        tqdm.write("Removing rows with missing price...")
        self.data = self.data.dropna(subset=['price'])

        # Filter outliers
        tqdm.write("Filtering outliers...")
        self.data = self.data[(self.data['price'] > 500) & (self.data['price'] < 100000)]
        self.data = self.data[(self.data['year'] > 1980)]
        self.data = self.data[(self.data['odometer'] < 400000)]

        print(f"Cleaning complete. {len(self.data)} rows remaining")
        return self
    
    def save_clean_data(self, output_filepath):
        """Saves the cleaned data to the filepath inserted on save"""
        if self.data is None:
            raise ValueError("Data not loaded or cleaned")
        
        self.data.to_csv(output_filepath, index=False)

        return self
    
    def get_data_split(self, target='price', test_size=0.2):
        """Creates the X and y data and split it"""
        X = self.data.drop(columns=[target])
        y = self.data[target]

        return train_test_split(X, y, test_size=test_size)