import pandas as pd
import logging
import json
import great_expectations as gx
from abc import ABC, abstractmethod
from great_expectations.core.expectation_suite import ExpectationSuite
from sqlalchemy import create_engine

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
        self,
        df: pd.DataFrame,
        column: str,
        rules_json_str: str,
        open_docs: bool = False,
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

    def __init__(self, datasource_config: dict = None):
        logger.info("Initializing GreatExpectationsValidator...")
        self.engine = None
        self.reference_data_cache = {}

        if datasource_config and datasource_config.get("type") in [
            "postgresql",
            "mysql",
            "sqlite",
        ]:
            try:
                db_type = datasource_config.get("type")
                user = datasource_config.get("user")
                host = datasource_config.get("host")
                port = datasource_config.get("port", 5432)
                db = datasource_config.get("db")
                password = datasource_config.get("password", "")

                connection_url = f"{db_type}://{user}:{password}@{host}:{port}/{db}"
                self.engine = create_engine(connection_url)
                logger.info("Validator created DB engine for FK checks.")
            except Exception as e:
                logger.warning(f"Validator failed to create DB engine: {e}")

    def _get_reference_data(self, query: str, column: str) -> pd.Series:
        cache_key = f"{query}_{column}"

        if cache_key in self.reference_data_cache:
            logger.info(f"Using cached reference data for: {cache_key}")
            return self.reference_data_cache[cache_key]

        if not self.engine:
            logger.error("DB Engine not available. Cannot fetch reference data.")
            raise ValueError("Validator DB Engine not initialized for FK check.")

        logger.info(f"Fetching reference data: {query[:100]}...")
        with self.engine.connect() as conn:
            ref_df = pd.read_sql(query, conn)

        if column not in ref_df.columns:
            logger.error(f"Reference column '{column}' not found in query result.")
            raise ValueError(f"Reference column '{column}' not found.")

        ref_series = ref_df[column]
        self.reference_data_cache[cache_key] = ref_series
        return ref_series

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

                if et == "expect_column_values_to_exist_in_other_table":
                    logger.debug("Processing FK rule...")
                    ref_query = kwargs.get("other_table_query")
                    ref_column = kwargs.get("other_table_column")

                    if ref_query and ref_column:
                        ref_data_series = self._get_reference_data(
                            ref_query, ref_column
                        )

                        del kwargs["other_table_query"]
                        del kwargs["other_table_column"]
                        kwargs["other_table_data"] = ref_data_series
                    else:
                        logger.warning("FK rule is missing query or column. Skipping.")
                        continue

                ExpClass = _snake_to_expect_class(et)
                exp_obj = ExpClass(**kwargs)
                suite.add_expectation(exp_obj)
                added += 1
            except Exception as e:
                logger.warning(f"Failed to add rule {rule}. Error: {e}")
                pass

        context.suites.add_or_update(suite)

        results = batch.validate(expect=suite)

        try:
            if open_docs:
                logger.info("Opening Data Docs in browser...")
                context.build_data_docs()
                context.open_data_docs()
            elif results["statistics"]["unsuccessful_expectations"] > 0:
                logger.info("Building Data Docs due to failures...")
                context.build_data_docs()
        except Exception as e:
            logger.warning(f"Failed to build or open Data Docs: {e}")

        return results if isinstance(results, dict) else results.to_json_dict()
