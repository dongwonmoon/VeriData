import pandas as pd
import logging

from .profilers import BaseProfiler
from .suggesters import BaseSuggester
from .validators import BaseValidator

logger = logging.getLogger(__name__)


class VeriDataPipeline:
    def __init__(
        self,
        profiler: BaseProfiler,
        suggester: BaseSuggester,
        validator: BaseValidator,
    ):
        logger.info("Initializing VeriDataPipeline...")
        self.profiler = profiler
        self.suggester = suggester
        self.validator = validator

    def run(
        self, df: pd.DataFrame, column: str, open_docs: bool = False
    ) -> dict:
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
