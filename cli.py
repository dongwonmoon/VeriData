import pandas as pd
import logging
import json
import typer
import yaml

from veridata.profilers import PandasProfiler
from veridata.suggesters import (
    OllamaRuleSuggester,
    OllamaDocSuggester,
    OpenAIRuleSuggester,
)
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


app = typer.Typer()


@app.command()
def run(
    config_path: str = typer.Option(
        "config.yml",
        "--config",
        "-c",
        help="Path to the VeriData config.yml file.",
    ),
    open_docs: bool = typer.Option(
        False,
        "--open",
        help="Open the Great Expectations HTML report in a browser after validation.",
    ),
):
    """
    VeriData 파이프라인을 config.yml 파일을 기반으로 실행합니다.
    """
    logger.info(f"Loading configuration from: {config_path}")

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        raise typer.Exit(code=1)

    logger.info(f"Config loaded: {config}")

    if config["components"]["profiler"] == "pandas":
        profiler = PandasProfiler()
    else:
        raise ValueError(f"Unknown profiler: {config['components']['profiler']}")

    suggester_config = config["components"]["suggester"]
    suggester_type = suggester_config.get("type")
    suggester_model = suggester_config.get("model")

    if suggester_type == "ollama":
        suggester = OllamaRuleSuggester(
            model=suggester_model or "gemma3:1b"
        )  # 기본값 설정
    elif suggester_type == "openai":
        if not suggester_model:
            suggester_model = "gpt-4o-mini"  # OpenAI 기본값
        suggester = OpenAIRuleSuggester(model=suggester_model)
    else:
        raise ValueError(f"Unknown suggester type: {suggester_type}")

    if config["components"]["validator"] == "great_expectations":
        validator = GreatExpectationsValidator()
    else:
        raise ValueError(f"Unknown validator: {config['components']['validator']}")

    pipeline = VeriDataPipeline(
        profiler=profiler, suggester=suggester, validator=validator
    )

    df_path = config["datasource"]["path"]
    df = load_data(df_path)
    if df.empty:
        logger.error("DataFrame is empty, pipeline stopped.")
        raise typer.Exit(code=1)

    columns = config["columns_to_validate"]
    logger.info(f"Target columns: {columns}")

    results = {}
    for col in columns:
        logger.info(f"--- Processing column: {col} ---")
        col_result = pipeline.run(df=df, column=col, open_docs=open_docs)
        results[col] = col_result
        logger.info(f"--- Finished column: {col} ---")

    logger.info("=== All columns processed ===")
    print("\n--- Final Validation Report ---")
    print(json.dumps(results, indent=2))


@app.command()
def document(
    config_path: str = typer.Option(
        "config.yml",
        "--config",
        "-c",
        help="Path to the VeriData config.yml file.",
    )
):
    logger.info(f"Loading configuration from: {config_path}")

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        raise typer.Exit(code=1)

    if config["components"]["profiler"] == "pandas":
        profiler = PandasProfiler()
    else:
        raise ValueError(f"Unknown suggester: {config['components']['suggester']}")

    doc_suggester = OllamaDocSuggester()

    df_path = config["datasource"]["path"]
    df = load_data(df_path)
    if df.empty:
        logger.error("DataFrame is empty, pipeline stopped.")
        raise typer.Exit(code=1)

    columns = config["columns_to_validate"]
    logger.info(f"Target columns for documentation: {columns}")

    print("\n--- VeriData Auto-Documentation ---")

    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found. Skipping.")
            continue

        logger.info(f"Generating doc for column: {col} ...")

        profile = profiler.profile(df=df, column=col)

        if not profile:
            logger.warning(f"Could not generate profile for '{col}'. Skipping.")
            continue

        description = doc_suggester.suggest(profile)

        print(f"\n## {col}")
        print(description)

    print("\n--- Documentation Generation Finished ---")


if __name__ == "__main__":
    app()
