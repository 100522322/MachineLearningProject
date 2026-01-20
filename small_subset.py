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
    
    print(f"Original data loaded. X shape: {X.shape}, y_reg shape: {y_reg.shape}, y_clf shape: {y_clf.shape}")

    # --- Subset creation ---
    subset_size = 5000  # Adjust as needed
    print(f"Creating a subset of {subset_size} samples...")
    
    # Use random indices to create a representative subset
    indices = np.random.choice(X.shape[0], subset_size, replace=False)
    
    X_subset = X[indices]
    y_reg_subset = y_reg[indices]
    y_clf_subset = y_clf[indices]
    
    print(f"Subset created. X shape: {X_subset.shape}, y_reg shape: {y_reg_subset.shape}, y_clf shape: {y_clf_subset.shape}")

    manager = ModelManager(r_state=42)

    # Try to load tuned parameters
    params_path = "./metrics/tuned_params.json"
    tuned_params = manager.load_params(params_path)
    if tuned_params:
        manager.set_params(tuned_params)
    else:
        print("No tuned parameters found. Using default hyperparameters.")
    
    # Train and test models using cross-validation on the subset
    print("Training and testing models on subset...")
    manager.train_test_models(X_subset, y_clf_subset, y_reg_subset)
    
    # Plot results
    print("Plotting results...")
    manager.plot_cv_results()

if __name__ == "__main__":
    main()