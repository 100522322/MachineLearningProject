"Script to preprocess data and save it to numpy files."
from config import CLEAN_DATA_FILE_PATH, X_PREPROCESSED_FILE_PATH, Y_PREPROCESSED_FILE_PATH
from src.preprocessor import Preprocessor

preprocessing = Preprocessor(CLEAN_DATA_FILE_PATH)
preprocessing.load_data()
preprocessing.preprocess_data()
preprocessing.save_preprocessed_data(X_PREPROCESSED_FILE_PATH, Y_PREPROCESSED_FILE_PATH)
