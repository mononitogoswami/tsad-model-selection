from yaml import load, dump
from yaml import CLoader as Loader, CDumper as Dumper

class Config:
    def __init__(
            self,
            config_file_path='../config.yaml'):
        """Class to read and parse the config.yml file
		"""
        self.config_file_path = config_file_path

    def parse(self):
        with open(self.config_file_path, 'rb') as f:
            self.config = load(f, Loader=Loader)
        return self.config

    def save_config(self):
        with open(self.config_file_path, 'w') as f:
            dump(self.config, f)