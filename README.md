# Final Project MAchine Learning course

Kaggle db: https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data


## How to run

1. Download the dataset from: https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data and insert on a /data folder
2. Run the data cleaning script: `python -m scripts.data_clean_script`
3. Run the data preprocessing script: `python -m scripts.data_preprocessing`
4. Run the model tuning script: `python -m scripts.tune_models`
5. Run the main script: `python main.py`

## How was constructed
Different classes were created to handle different tasks:

- `DataLoader`: Loads the data from the CSV file
- `Preprocessor`: Preprocesses the data
- `ModelManager`: Manages the training and testing of models

### ModelManager
We created a class ModelManager that handles the training and testing of models. It has the following methods:

- `train_test_models`: Trains and tests models using cross-validation
- `tune_models`: Tunes models using grid search
- `plot_cv_results`: Plots the results of the cross-validation
- `save_model`: Saves a model to a file
- `load_model`: Loads a model from a file
- `save_params`: Saves the parameters of the models to a file
- `load_params`: Loads the parameters of the models from a file
- `save_results_json`: Saves the results of the cross-validation to a file
- `load_results_json`: Loads the results of the cross-validation from a file

### DataLoader
This class is responsible for loading the data from the CSV file and saving the clean data to a new CSV file.
It has the following methods:

- `load_data`: Loads the data from the CSV file
- `clean_data`: Cleans the data
- `save_clean_data`: Saves the clean data to a new CSV file

### Preprocessor
This class is responsible for preprocessing the data. It uses the `ColumnTransformer` class to preprocess the data.
It has the following methods:

- `load_data`: Searches and loads data from the path of innitial data
- `preprocess_data`: Preprocesses the data
- `save_preprocessed_data`: Saves the preprocessed data to a .npy file

### Scripts
- `data_clean_script.py`: Cleans the data
- `data_preprocessing.py`: Preprocesses the data
- `tune_models.py`: Tunes the models and saves the parameters
- `main.py`: Main script to run the models
- `small_subset.py`: Makes the same as main.py but with a small subset of the data for testing the code

## Results
Once the code is run, the following results are generated:
