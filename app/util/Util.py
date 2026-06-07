import time
from datetime import datetime
from pathlib import Path
from string import Template


# This util should not rely on any self defined class or method.
# To make it pure, it can only rely on python itself or third party library


class Util(object):

    @staticmethod
    def get_base_path(reference_file: str) -> Path:
        """Return the absolute path to the root directory path."""
        current_path = Path(__file__).resolve()

        for parent in [current_path, *current_path.parents]:
            if parent.joinpath(reference_file).exists():
                return parent.resolve()

        raise ValueError(f"Could not find base path for reference file: {reference_file}")

    @staticmethod
    def strip_string(value):
        result = value
        if result and type(result) == str:
            result = result.strip()
        return result
