from abc import ABC, abstractmethod
import pandas as pd
import logging
import numpy as np
import datetime

logger = logging.getLogger(__name__)


class BaseProfiler(ABC):
    """
    Abstract base class for data profilers.
    """

    @abstractmethod
    def profile(self, df: pd.DataFrame, column: str) -> dict:
        """
        Profiles a column in a DataFrame and returns a dictionary of profile information.

        Args:
            df (pd.DataFrame): The DataFrame to profile.
            column (str): The column to profile.

        Returns:
            dict: A dictionary of profile information.
        """
        pass


class PandasProfiler(BaseProfiler):
    """
    A data profiler that uses pandas to generate profile information.
    """

    def __init__(self):
        pass

    def _serialize_value(self, val):
        """Helper to safely serialize profile values for JSON."""
        if pd.isna(val):
            return None

        if isinstance(val, (datetime.date, datetime.datetime, pd.Timestamp)):
            return val.isoformat()

        if isinstance(val, np.generic):
            return val.item()

        if isinstance(val, (int, float, str, bool)):
            return val

        logger.warning(f"Could not serialize type {type(val)}, converting to str.")
        return str(val)

    def profile(self, df: pd.DataFrame, column: str) -> dict:
        """
        Profiles a column in a DataFrame and returns a dictionary of profile information.

        Args:
            df (pd.DataFrame): The DataFrame to profile.
            column (str): The column to profile.

        Returns:
            dict: A dictionary of profile information.
        """
        logger.info(f"Profiling column '{column}'...")
        if column not in df.columns:
            logger.error(f"Column '{column}' not found in DataFrame.")
            return {}

        series = df[column]
        dtype = series.dtype
        min_value = None
        max_value = None
        if pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_datetime64_any_dtype(
            dtype
        ):
            try:
                min_value = series.min()
                max_value = series.max()
            except TypeError as e:
                logger.warning(f"Could not get min/max for column {column}: {e}")
        top_5_samples_raw = series.value_counts().head(5).index.tolist()

        profile = {
            "column_name": column,
            "data_type": str(dtype),
            "null_percentage": (series.isnull().mean() * 100).item(),
            "unique_values_count": series.nunique(),
            "min": self._serialize_value(min_value),
            "max": self._serialize_value(max_value),
            "top_5_samples": [self._serialize_value(s) for s in top_5_samples_raw],
        }
        logger.info(f"Profile for '{column}': {profile}")
        return profile
