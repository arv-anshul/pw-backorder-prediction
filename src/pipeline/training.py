from dataclasses import dataclass

import dataframe_image as dfi
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from src.data.data_transformer import DataTransformer
from src.logger import logging


@dataclass
class ModelTrainer:
    """ Train ML models and evaluate. """

    X_train: list
    X_test: list
    y_train: list
    y_test: list
    transformer: DataTransformer

    def evaluate_models_report(self, models: dict, params: dict | None = None,
                               save_score_report: bool = False):
        """
        Train model with cross validation technique using `GridSearchCV` and 
        evaluate with `accuracy_score`.
        """

        # Get transformed array data
        (X_train_trf,
         X_test_trf) = self.transformer.transform(self.X_train, self.X_test)
        logging.info('Transformed: X_train, X_test')

        report = {}

        for model_name, model_obj in models.items():
            param = params[model_name] if params else {}

            # Train GridSearchCV algorithm
            gs = GridSearchCV(model_obj, param, cv=3)
            gs.fit(X_train_trf, self.y_train)
            logging.info(f'Model Fit: {model_name}')

            # Predict X_test
            y_test_pred = gs.predict(X_test_trf)    # type: ignore

            # Calculate accuracy_score of trained model
            score = accuracy_score(self.y_test, y_test_pred)
            logging.info(f'accuracy_score of {model_name} is {score}.')

            # Store score in dictionary
            report[model_name] = score

        report = dict(sorted(report.items(), key=lambda x: x[1]))

        if save_score_report:
            report_df = pd.DataFrame({
                'Models': report.keys(), 'accuracy_score': report.values()
            })
            dfi.export(report_df, 'report_df.png')

        return report

    def get_best_model(self, report: dict[str, float]):
        best_model = list(report.items())[0]

        if best_model[1] < 0.7:
            raise ValueError(
                f'Best model has only {best_model[1]:.2f} accuracy_score. '
                'Try different models.'
            )
        return best_model
