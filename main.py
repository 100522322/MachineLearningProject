from src.model_manager import ModelManager
from config import X_PREPROCESSED_FILE_PATH, Y_PREPROCESSED_FILE_PATH
import numpy as np

def main():
    
    print("Loading data...")
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
    
    # Train and test models using cross-validation
    print("Training and testing models...")
    manager.train_test_models(X, y_clf, y_reg)
    
    # Plot results
    print("Plotting results...")
    manager.plot_cv_results()

if __name__ == "__main__":
    main()