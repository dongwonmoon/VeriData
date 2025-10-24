from abc import ABC, abstractmethod
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class BaseProfiler(ABC):
    @abstractmethod
    def profile(self, df: pd.DataFrame, column: str) -> dict:
        pass


class PandasProfiler(BaseProfiler):
    def __init__(self):
        pass

    def profile(self, df: pd.DataFrame, column: str) -> dict:
        logger.info(f"Profiling column '{column}'...")
        if column not in df.columns:
            logger.error(f"Column '{column}' not found in DataFrame.")
            return {}

        series = df[column]
        dtype = series.dtype
        min_value = None
        max_value = None
        if pd.api.types.is_numeric_dtype(dtype):
            min_value = series.min()
            max_value = series.max()
        top_5_samples = series.value_counts().head(5).index.tolist()

        profile = {
            "column_name": column,
            "data_type": str(dtype),
            "null_percentage": (series.isnull().mean() * 100).item(),
            "unique_values_count": series.nunique(),
            "min": min_value.item() if min_value is not None else None,
            "max": max_value.item() if max_value is not None else None,
            "top_5_samples": top_5_samples,
        }
        logger.info(f"Profile for '{column}': {profile}")
        return profile
