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
    
    # Try to load tuned parameters
    params_path = "./metrics/tuned_params.json"
    tuned_params = manager.load_params(params_path)
    if tuned_params:
        manager.set_params(tuned_params)
    else:
        print("No tuned parameters found. Using default hyperparameters. Run scripts/tune_models.py to tune.")

    # Train and test models using cross-validation
    print("Training and testing models...")
    manager.train_test_models(X, y_clf, y_reg, splits_n=10)
    
    # Plot results
    print("Plotting results...")
    manager.plot_cv_results()

if __name__ == "__main__":
    main()