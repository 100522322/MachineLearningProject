"""Script for cleaning and saving data"""
from config import DATA_FILE_PATH, CLEAN_DATA_FILE_PATH
from src.data_loader import DataLoader


data_loader = DataLoader(DATA_FILE_PATH)
data_loader.load_data()
data_loader.clean_data()
data_loader.save_clean_data(CLEAN_DATA_FILE_PATH)


