from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class DataAccessor:
    """ Path of data in parquet format. """
    cleaned_data_path: Path
    target_col_name: str

    def __post_init__(self):
        # Check clean data file type
        if self.cleaned_data_path.suffix != '.parquet':
            raise ValueError("Cleaned data file extension must be '.parquet'")

        self.df = pd.read_parquet(self.cleaned_data_path)

    def get_cat_and_num_cols(self):
        """ Returns (cat_cols, num_cols) """
        columns = self.df.columns.drop(self.target_col_name, 'ignore')

        cat_cols = [i for i in columns if self.df[i].dtype == 'O']
        num_cols = [i for i in columns if self.df[i].dtype != 'O']
        return cat_cols, num_cols

    def best_corr_cols(self, df: pd.DataFrame, top_n_cols: int) -> list[str]:
        corr_matrix = df.corr()
        target_col_corr = corr_matrix[self.target_col_name]
        best_corr_cols = list(
            target_col_corr
            .apply(lambda x: abs(x))
            .sort_values(ascending=False)
            [:top_n_cols].index
        )
        return best_corr_cols

    def get_train_test_df(self, test_size=0.2, random_state=42,):
        X = self.df.drop(columns=[self.target_col_name])
        y = self.df[self.target_col_name]

        return train_test_split(X, y,
                                test_size=test_size,
                                random_state=random_state)
