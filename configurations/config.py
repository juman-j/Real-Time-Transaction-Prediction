from pathlib import Path

from configurations.helpers.JSONReader import JSONReader

cwd = Path.cwd()
config = JSONReader.initialization_of_basic_configuration(cwd)
