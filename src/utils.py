""" Extra functions for this project. """

from pathlib import Path


class ImportantPaths:
    """ Paths for the project. """

    def __init__(self, root_path: Path) -> None:
        self.root = root_path

    def processed_data_path(self, processed_data_filename: str):
        return Path(self.root, 'data/processed', processed_data_filename)

    def trained_model_path(self, model_filename: str):
        return Path(self.root, 'data/model', model_filename)
