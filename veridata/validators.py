import pandas as pd
import logging
import json
import great_expectations as gx
from abc import ABC, abstractmethod
from great_expectations.core.expectation_suite import ExpectationSuite

logger = logging.getLogger(__name__)


def _snake_to_expect_class(expectation_type: str):
    """
    Converts a snake_case expectation type string to a Great Expectations class name.

    Args:
        expectation_type (str): The snake_case expectation type string.

    Returns:
        The Great Expectations class.
    """
    parts = expectation_type.strip().split("_")
    if parts[0] != "expect":
        parts.insert(0, "expect")
    parts[0] = "Expect"
    class_name = "".join(p.capitalize() for p in parts)
    try:
        return getattr(gx.expectations, class_name)
    except AttributeError:
        raise ValueError(
            f"Unknown expectation class for '{expectation_type}' -> '{class_name}'"
        )


class BaseValidator(ABC):
    """
    Abstract base class for data validators.
    """

    @abstractmethod
    def validate(
        self, df: pd.DataFrame, column: str, rules_json_str: str
    ) -> dict:
        """
        Validates a DataFrame column based on a set of rules.

        Args:
            df (pd.DataFrame): The DataFrame to validate.
            column (str): The column to validate.
            rules_json_str (str): A JSON string containing the validation rules.

        Returns:
            dict: The validation results.
        """
        pass


class GreatExpectationsValidator(BaseValidator):
    """
    A data validator that uses Great Expectations to validate data.
    """

    def __init__(self):
        pass

    def validate(
        self,
        df: pd.DataFrame,
        column: str,
        rules_json_str: str,
        open_docs: bool = False,
    ) -> dict:
        """
        Validates a DataFrame column using Great Expectations.

        Args:
            df (pd.DataFrame): The DataFrame to validate.
            column (str): The column to validate.
            rules_json_str (str): A JSON string containing the validation rules.
            open_docs (bool, optional): Whether to open the validation report in a browser. Defaults to False.

        Returns:
            dict: The validation results.
        """
        logger.info(f"Validating data for {column}...")
        try:
            rules_list = json.loads(rules_json_str)
            if not isinstance(rules_list, list):
                return {
                    "success": False,
                    "error": "LLM did not return a JSON list.",
                }
        except json.JSONDecodeError as e:
            return {"success": False, "error": f"Error decoding JSON: {e}"}

        context = gx.get_context()

        ds = context.data_sources.add_pandas(name="pandas_ds")
        asset = ds.add_dataframe_asset(name="my_dataframe_asset")
        bd = asset.add_batch_definition_whole_dataframe(name="whole_df")

        batch = bd.get_batch(batch_parameters={"dataframe": df})

        suite_name = f"suite_for_{column}"
        try:
            suite = context.suites.get(suite_name)
        except Exception:
            suite = ExpectationSuite(name=suite_name)
            context.suites.add(suite)

        added = 0
        for rule in rules_list:
            try:
                et = rule["expectation_type"]
                kwargs = rule.get("kwargs", {})
                ExpClass = _snake_to_expect_class(et)
                exp_obj = ExpClass(**kwargs)
                suite.add_expectation(exp_obj)
                added += 1
            except Exception as e:
                pass

        context.suites.add_or_update(suite)

        results = batch.validate(expect=suite)

        try:
            logger.info("Building Data Docs (HTML Report)")

            context.build_data_docs()

            if open_docs:
                logger.info("Opening Data Docs (HTML Report)")
                context.open_data_docs()
        except Exception as e:
            logger.warning(f"Failed to build or open Data Docs: {e}")

        return results if isinstance(results, dict) else results.to_json_dict()
