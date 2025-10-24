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
from veridata.pipeline import VeriDataPipeline


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

    pipeline = VeriDataPipeline(
        profiler=profiler, suggester=suggester, validator=validator
    )

    if not df.empty:
        validation_result = pipeline.run(df=df, column="age")

        print(f"Validation Result: {validation_result}")
    else:
        logger.warning("DataFrame is empty, skipping validation.")
