import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
import json

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score, r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor, XGBClassifier
from sklearn.base import clone


class ModelManager:

    def __init__(self, r_state=None):
        """
        {"NAME": CLASS()}
        """
        self.models_reg = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0),
            "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
            "Random Forest": RandomForestRegressor(
                n_estimators=50,
                max_depth=15,
                n_jobs=-1,
                random_state=42
            ),
            "XGBoost": XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=10,
                n_jobs=-1,
                random_state=42
            )
        }

        self.models_clf = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest Clf": RandomForestClassifier(
                n_estimators=50, 
                max_depth=15, 
                n_jobs=-1, 
                random_state=42
            ),
            "XGBoost Clf": XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=10,
                n_jobs=-1,
                random_state=42
            )
        }
        self.results = None

    def save_model(self, model, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model, path)

    def load_model(self, path):
        return joblib.load(path)

    def _to_json_safe(self, obj):
        if isinstance(obj, dict):
            return {k: self._to_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._to_json_safe(v) for v in obj]
        elif isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        else:
            return obj

    def save_results_json(self, results, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        safe_results = self._to_json_safe(results)

        with open(path, "w") as f:
            json.dump(safe_results, f, indent=4)

        print(f"Results saved to {path}")

    def train_test_models(self, X, y_clf, y_reg, splits_n=5):
        cv = KFold(n_splits=splits_n, shuffle=True, random_state=42)
        raw_results = {
            "Regressors":{
                name: {"MAE": [], "RMSE": [], "R2":[]} for name in self.models_reg
            },
            "Classifiers":{
                name: {"Accuracy": [], "F1": []} for name in self.models_clf
            }

        }

        print("Starting Cross-Validation training...")
        for fold, (train_index, test_index) in enumerate(cv.split(X)):
            X_train, X_test = X[train_index], X[test_index]
            y_reg_train, y_reg_test = y_reg[train_index], y_reg[test_index]
            y_clf_train, y_clf_test = y_clf[train_index], y_clf[test_index]


            print(f"=== Fold {fold+1}/{splits_n} ===")
            print(f"X_train: {X_train.shape}")

            # -------------Regressors----------------
            print("\tRegressors:")
            for name, model in self.models_reg.items():
                print(f"\t\t{name}")
                model_fold = clone(model)
                model_fold.fit(X_train, y_reg_train)
                y_pred = model_fold.predict(X_test)

                mae = mean_absolute_error(y_reg_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred))
                r2 = r2_score(y_reg_test, y_pred)

                raw_results["Regressors"][name]["MAE"].append(mae)
                raw_results["Regressors"][name]["RMSE"].append(rmse)
                raw_results["Regressors"][name]["R2"].append(r2)

                """
                if fold == splits_n - 1:
                    print(f"Saving regressor model: {name}")
                    path = f"./models/{name}_last_fold.joblib"
                    self.save_model(model_fold, path)
                """

            # ------------Classifiers---------------
            print("\tClassifiers:")
            for name, model in self.models_clf.items():
                print(f"\t\t{name}")
                model_fold = clone(model)
                model_fold.fit(X_train, y_clf_train)

                y_pred = model_fold.predict(X_test)

                acc = accuracy_score(y_clf_test, y_pred)
                f1 = f1_score(y_clf_test, y_pred, average='weighted')

                raw_results["Classifiers"][name]["Accuracy"].append(acc)
                raw_results["Classifiers"][name]["F1"].append(f1)

                """
                if fold == splits_n - 1:
                    print(f"Saving classifier model: {name}")
                    path = f"./models/{name}_last_fold.joblib"
                    self.save_model(model_fold, path)
                """

        # Process results to get mean and std
        processed_results = {"Regressors": {}, "Classifiers": {}}
        for group, models in raw_results.items():
            for name, metrics in models.items():
                processed_results[group][name] = {
                    metric:{
                        "mean": np.mean(values),
                        "std": np.std(values)
                    }
                    for metric, values in metrics.items()
                }



        print("Finished Cross-Validation.")

        self.save_results_json(processed_results, "./metrics/procesed_results.json")
        self.save_results_json(processed_results, "./metrics/raw_results.json")
        self.processed_results = processed_results
        self.raw_results = raw_results
        return processed_results

    def plot_cv_results(self, results=None):
        """
        Plot cross-validation results using matplotlib.
        Prioritizes Box Plots if raw_results are available, otherwise uses Bar Plots.
        """
        # 1. Try to use raw_results for Box Plots (Better for CV)
        if hasattr(self, 'raw_results') and self.raw_results is not None:
            print("Plotting Box Plots using raw cross-validation results...")
            for group_name, models in self.raw_results.items():
                if not models:
                    continue

                metrics = list(next(iter(models.values())).keys())
                n_metrics = len(metrics)
                
                # Create subplots
                fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 6 * n_metrics))
                if n_metrics == 1:
                    axes = [axes]

                for ax, metric in zip(axes, metrics):
                    data = []
                    labels = []
                    
                    for name, model_metrics in models.items():
                        labels.append(name)
                        data.append(model_metrics[metric])
                    
                    # Box Plot
                    ax.boxplot(data, tick_labels=labels, patch_artist=True)
                    ax.set_title(f"{group_name} - {metric} Distribution (CV Folds)")
                    ax.set_ylabel(metric)
                    ax.grid(True, linestyle='--', alpha=0.6)
                    
                    # Rotate x-labels if needed
                    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

                fig.tight_layout()
                save_path = f"./metrics/{group_name}_performance_boxplot.png"
                plt.savefig(save_path)
                plt.close(fig) # Close to free memory
                print(f"Saved plot to {save_path}")
            return

        # 2. Fallback to Bar Plots if only processed results are available
        print("Raw results not found. Plotting Bar Charts using aggregated results...")
        if results is None:
            results = self.processed_results
            if results is None:
                print("No results to plot.")
                return

        for group_name, models in results.items():
            if not models:
                continue

            # Get metrics from the first model of the group
            metrics = list(next(iter(models.values())).keys())
            n_metrics = len(metrics)
            n_models = len(models)

            fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 5 * n_metrics), sharex=False)
            if n_metrics == 1:
                axes = [axes]

            for ax, metric in zip(axes, metrics):
                model_names = list(models.keys())
                mean_values = [models[name][metric]["mean"] for name in model_names]
                std_values = [models[name][metric]["std"] for name in model_names]

                x_pos = np.arange(n_models)
                bars = ax.bar(x_pos, mean_values, yerr=std_values, capsize=5, alpha=0.7)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(model_names, rotation=45, ha="right")
                ax.set_ylabel(metric)
                ax.set_title(f"Average {metric} for {group_name}")
                ax.grid(axis='y', linestyle='--')

                # Add values on top of bars
                for bar in bars:
                    yval = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f}', va='bottom', ha='center')


            fig.suptitle(f'{group_name} Performance Comparison (Mean +/- Std)', fontsize=16, y=1.02)
            plt.tight_layout()
            save_path = f"./metrics/{group_name}_performance.png"
            plt.savefig(save_path)
            plt.close(fig)
            print(f"Saved plot to {save_path}")
