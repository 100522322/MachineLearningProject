import sys
import os
import numpy as np

# Add project root to path to allow imports from src and config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model_manager import ModelManager
from config import X_PREPROCESSED_FILE_PATH, Y_PREPROCESSED_FILE_PATH

def main():
    print("Loading data for tuning...")
    try:
        X = np.load(X_PREPROCESSED_FILE_PATH)
        y = np.load(Y_PREPROCESSED_FILE_PATH, allow_pickle=True)
    except FileNotFoundError:
        print("Preprocessed data not found. Please run scripts/data_preprocessing.py first.")
        return

    # Extract y_reg and y_clf from y
    # y[0] is price (regression), y[1] is price category (classification)
    y_reg = y[0]
    y_clf = y[1]
    
    print(f"Data loaded. X shape: {X.shape}, y_reg shape: {y_reg.shape}, y_clf shape: {y_clf.shape}")

    manager = ModelManager(r_state=42)
    
    # Run tuning
    print("Starting model tuning. This may take a while...")
    best_params = manager.tune_models(X, y_clf, y_reg)
    
    # Save tuned parameters
    params_path = "./metrics/tuned_params.json"
    manager.save_params(best_params, params_path)
    print("Tuning finished.")

if __name__ == "__main__":
    main()
