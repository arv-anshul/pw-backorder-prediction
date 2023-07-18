from pathlib import Path

from src import utils
from src.config import TARGET_COLUMN
from src.data import DataAccessor, DataTransformer
from src.pipeline import ModelTrainer

imp_path = utils.ImportantPaths(Path('.'))

data_accessor = DataAccessor(imp_path.processed_data_path('cleaned_back_order_data_5000.parquet'),
                             TARGET_COLUMN)

def data_transformation_pipeline():
    transformer = DataTransformer(*data_accessor.get_cat_and_num_cols())
    return transformer


def model_training_pipeline(X_train_trf, X_test_trf, y_train, y_test,
                            transformer_obj):
    trainer = ModelTrainer(X_train_trf, X_test_trf,    # type: ignore
                           y_train, y_test, transformer=transformer_obj)
