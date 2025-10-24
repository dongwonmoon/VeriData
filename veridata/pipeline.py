import pandas as pd
import logging

from .profilers import BaseProfiler
from .suggesters import BaseSuggester
from .validators import BaseValidator

logger = logging.getLogger(__name__)


class VeriDataPipeline:
    """
    The main pipeline for Veri-Data. It orchestrates the profiling, suggestion, and validation steps.
    """

    def __init__(
        self,
        profiler: BaseProfiler,
        suggester: BaseSuggester,
        validator: BaseValidator,
    ):
        """
        Initializes the VeriDataPipeline.

        Args:
            profiler (BaseProfiler): An instance of a data profiler.
            suggester (BaseSuggester): An instance of a rule suggester.
            validator (BaseValidator): An instance of a data validator.
        """
        logger.info("Initializing VeriDataPipeline...")
        self.profiler = profiler
        self.suggester = suggester
        self.validator = validator

    def run(
        self, df: pd.DataFrame, column: str, open_docs: bool = False
    ) -> dict:
        """
        Runs the pipeline for a given DataFrame and column.

        Args:
            df (pd.DataFrame): The DataFrame to process.
            column (str): The column to process.
            open_docs (bool, optional): Whether to open the validation report in a browser. Defaults to False.

        Returns:
            dict: The validation results.
        """
        logger.info(f"Running VeriDataPipeline for column '{column}'...")

        profile = self.profiler.profile(df, column)
        if profile:
            suggested_rules = self.suggester.suggest(profile)
            validation_result = self.validator.validate(
                df, column, suggested_rules, open_docs=open_docs
            )
            return validation_result
        else:
            logger.warning("Profile is empty, skipping rule suggestion.")
            return {}
