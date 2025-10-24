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
from veridata.validators import GreatExpectationsValidator


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


if __name__ == "__main__":
    df = load_data("./sample.csv")
    profiler = PandasProfiler()
    suggester = OllamaRuleSuggester()
    validator = GreatExpectationsValidator()

    if not df.empty:
        profile = profiler.profile(df, "age")
        if profile:
            suggested_rules = suggester.suggest(profile)
            print("--- LLM Suggested Rules ---")
            print(suggested_rules)

            validation_result = validator.validate(df, "age", suggested_rules)
            print("\n--- Validation Results ---")
            print(json.dumps(validation_result, indent=2))
        else:
            logger.warning("Profile is empty, skipping rule suggestion.")
    else:
        logger.warning("DataFrame is empty, pipeline stopped.")
