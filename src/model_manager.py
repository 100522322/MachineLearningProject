import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
import json
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score
from sklearn.base import clone

class ModelManager:

    def __init__(self):
        """
        {"NAME": CLASS()}
        """
        self.models_reg = {

        }

        self.models_clf = {

        }
        self.results = None

    def save_model(self, model, path):
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
        tscv = TimeSeriesSplit(n_splits=splits_n)
        results = {
            "Regressors":{
                name: {"MAE": [], "RMSE": []}
                for name in self.models_reg
            },
            "Classifiers":{
                name: {"Accuracy": [], "F1": []}
                for name in self.models_clf
            }

        }

        print("Starting training...")
        for fold, (train_index, test_index) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_index], X[test_index]

            y_clf_train, y_clf_test = y_clf[train_index], y_clf[test_index]
            y_reg_train, y_reg_test = y_reg[train_index], y_reg[test_index]

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

                results["Regressors"][name]["MAE"].append(mae)
                results["Regressors"][name]["RMSE"].append(rmse)

                if fold == splits_n - 1:
                    print(f"Saving regressor model: {name}")
                    path = f"./models/{name}_last_fold.joblib"
                    self.save_model(model_fold, path)

            # ------------Classifiers---------------
            print("\tClassifiers:")
            for name, model in self.models_clf.items():
                print(f"\t\t{name}")
                model_fold = clone(model)
                model_fold.fit(X_train, y_clf_train)

                y_pred = model_fold.predict(X_test)

                acc = accuracy_score(y_clf_test, y_pred)
                f1 = f1_score(y_clf_test, y_pred)

                results["Classifiers"][name]["Accuracy"].append(acc)
                results["Classifiers"][name]["F1"].append(f1)

                if fold == splits_n - 1:
                    print(f"Saving regressor model: {name}")
                    path = f"./models/{name}_last_fold.joblib"
                    self.save_model(model_fold, path)


        print("Finished training.")

        self.save_results_json(results, "./metrics/results.json")
        self.results = results
        return results

    def train_final_model(self, X, y, model_name, save_path=None):
        pass


    def plot_cv_results(self, results=None):
        """
            Plot cross-validation results using matplotlib.
        """
        if results is None:
            results = self.results

        for group_name, models in results.items():
            if not models:
                continue

            # Get metrics from the first model of the group
            metrics = list(next(iter(models.values())).keys())
            n_metrics = len(metrics)

            fig, axes = plt.subplots(
                n_metrics, 1,
                figsize=(8, 4 * n_metrics),
                sharex=True
            )

            # If only one metric, axes is not iterable
            if n_metrics == 1:
                axes = [axes]

            for ax, metric in zip(axes, metrics):
                for model_name, model_metrics in models.items():
                    values = np.array(model_metrics[metric], dtype=float)
                    folds = np.arange(1, len(values) + 1)

                    ax.plot(folds, values, marker="o", label=model_name)

                ax.set_ylabel(metric)
                ax.set_title(f"{group_name} – {metric} per fold")
                ax.grid(True)
                ax.legend()

                # Optional: nicer limits for accuracy-like metrics
                if metric.lower() in ("accuracy", "f1"):
                    ax.set_ylim(0.45, 0.65)

            axes[-1].set_xlabel("Fold")

            fig.suptitle(group_name, fontsize=14)
            plt.tight_layout()
            plt.show()
            plt.savefig("./metrics/foo.png")
