import pandas as pd
import logging
import json
import great_expectations as gx
import shutil
import os
import re

from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from veridata.profilers import PandasProfiler
from veridata.suggesters import OllamaRuleSuggester


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_data(csv_path: str) -> pd.DataFrame:
    logger.info(f"S1: Loading data from {csv_path}")
    try:
        return pd.read_csv(csv_path)
    except FileNotFoundError:
        logger.error(f"File not found: {csv_path}")
        return pd.DataFrame()


def _snake_to_expect_class(expectation_type: str):
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


def validate_data(df, column: str, rules_json_str: str) -> dict:
    try:
        rules_list = json.loads(rules_json_str)
        if not isinstance(rules_list, list):
            return {"success": False, "error": "LLM did not return a JSON list."}
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
        from great_expectations.core.expectation_suite import ExpectationSuite

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

    return results if isinstance(results, dict) else results.to_json_dict()


if __name__ == "__main__":
    df = load_data("./sample.csv")
    profiler = PandasProfiler()
    suggester = OllamaRuleSuggester()

    if not df.empty:
        profile = profiler.profile(df, "age")
        if profile:
            suggested_rules = suggester.suggest(profile)
            print("--- LLM Suggested Rules ---")
            print(suggested_rules)

            validation_result = validate_data(df, "age", suggested_rules)
            print("\n--- Validation Results ---")
            print(json.dumps(validation_result, indent=2))
        else:
            logger.warning("Profile is empty, skipping rule suggestion.")
    else:
        logger.warning("DataFrame is empty, pipeline stopped.")
