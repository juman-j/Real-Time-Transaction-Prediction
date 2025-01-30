import json
import os
from pathlib import Path

from loguru import logger

from data_models.config_models import (
    Config,
    RandomForestHyperparameters,
    XGBoostHyperparameters,
    ModelParameters,
)


class JSONReader:
    @staticmethod
    def find_file(
        cwd_path: Path,
        file_name: str,
        directory: str = None,
        subdirectory: str = None,
    ) -> Path | None:
        if directory:
            file_path = Path(os.path.join(cwd_path, directory, file_name))
            if file_path.is_file():
                return file_path
            if subdirectory:
                file_path = Path(
                    os.path.join(cwd_path, directory, subdirectory, file_name)
                )
                if file_path.is_file():
                    return file_path

        file_path = Path(os.path.join(cwd_path, file_name))
        if file_path.is_file():
            return file_path

        for parant in cwd_path.parents:
            file_path = Path(os.path.join(parant, file_name))
            if file_path.is_file():
                return file_path
        logger.error(f"No {file_name} found.")
        return None

    @classmethod
    def __finde_and_read_json_config(cls, cwd: Path) -> dict | None:
        file_path = cls.find_file(
            cwd_path=cwd, file_name="config.json", directory="configurations"
        )
        if file_path:
            with open(file_path, encoding="UTF-8") as f:
                config: dict = json.load(f)
                assert config, "The 'config.json' is empty or incorrectly composed."
            return config
        return None

    @classmethod
    def __parse_config(cls, config_dict: dict) -> Config:
        rf_params_dict: dict = config_dict.get("RandomForest", {})
        xgb_params_dict: dict = config_dict.get("XGBoost", {})

        rf_params = RandomForestHyperparameters(**rf_params_dict)
        xgb_params = XGBoostHyperparameters(**xgb_params_dict)

        model_params = ModelParameters(RandomForest=rf_params, XGBoost=xgb_params)

        return Config(model_parameters=model_params)

    @classmethod
    def initialization_of_basic_configuration(cls, cwd: Path) -> Config:
        config_dict = cls.__finde_and_read_json_config(cwd)

        if config_dict:
            return cls.__parse_config(config_dict)
        else:
            return Config()
