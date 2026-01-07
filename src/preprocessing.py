from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def get_preprocessor():
    """
    Imputer for filling missing values
    StandardScaler for scaling numerical features
    OneHotEncoder for encoding categorical features
    """

    numeric_features = ['year', 'odometer']
    categorical_features = ['manufacturer', 'condition', 'cylinders', 'fuel',
                            'title_status', 'transmission', 'drive', 'size',
                            'type', 'paint_color']
    
    # Pipeline for numerical features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
        ])
    # Pipeline for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='consntant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
    
    # Combine both transformers into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    return preprocessor