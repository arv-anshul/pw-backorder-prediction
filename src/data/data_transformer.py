from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.logger import logging


@dataclass
class DataTransformer:
    """ Data transformer pipeline. """

    cat_cols: list[str]
    num_cols: list[str]

    def get_transformer(self):
        num_pipeline = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
            ]
        )

        cat_pipeline = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False)),
            ]
        )

        logging.info('Categorical Columns: %s', self.cat_cols)
        logging.info('Numerical Columns: %s', self.num_cols)

        transformer = ColumnTransformer(
            transformers=[
                ('cat_pipelines', cat_pipeline, self.cat_cols),
                ('num_pipeline', num_pipeline, self.num_cols),
            ]
        )
        return transformer

    def transform(self, X_train, X_test):
        transformer = self.get_transformer()
        logging.info('Accessed the transformer.')

        X_train_trf = transformer.fit_transform(X_train)
        logging.info('Transformed X_train.')

        X_test_trf = transformer.transform(X_test)
        logging.info('Transformed X_test.')

        return X_train_trf, X_test_trf
