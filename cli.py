import pandas as pd
import logging
import json
import typer
import yaml

from veridata.loaders import BaseLoader, CsvLoader, SqlLoader
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

app = typer.Typer()


def get_loader(datasource_config: dict) -> BaseLoader:
    """
    Initializes and returns a data loader based on the datasource configuration.

    Args:
        datasource_config (dict): The datasource configuration.

    Returns:
        BaseLoader: An instance of a data loader.
    """
    ds_type = datasource_config.get("type")
    logger.info(f"Initializing DataLoader (Type: {ds_type})...")

    if ds_type == "csv":
        return CsvLoader()
    elif ds_type in ["postgresql", "mysql", "sqlite"]:
        return SqlLoader()
    else:
        raise ValueError(f"Unknown datasource type: {ds_type}")


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
    Runs the VeriData pipeline based on the provided config.yml file.
    """
    logger.info(f"Loading configuration from: {config_path}")

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        raise typer.Exit(code=1)

    logger.info(f"Config loaded: {config}")

    all_relationships = config.get("relationships", [])
    logger.info(f"Loaded {len(all_relationships)} relationships from config.")
    datasource_config = config["datasource"]

    if config["components"]["profiler"] == "pandas":
        profiler = PandasProfiler()
    else:
        raise ValueError(f"Unknown profiler: {config['components']['profiler']}")

    suggester_config = config["components"]["suggester"]
    suggester_type = suggester_config.get("type")
    suggester_model = suggester_config.get("model")

    if suggester_type == "ollama":
        suggester = OllamaRuleSuggester(model=suggester_model or "gemma3:1b")
    elif suggester_type == "openai":
        if not suggester_model:
            suggester_model = "gpt-4o-mini"
        suggester = OpenAIRuleSuggester(model=suggester_model)
    else:
        raise ValueError(f"Unknown suggester type: {suggester_type}")

    if config["components"]["validator"] == "great_expectations":
        validator = GreatExpectationsValidator(datasource_config=datasource_config)
    else:
        raise ValueError(f"Unknown validator: {config['components']['validator']}")

    pipeline = VeriDataPipeline(
        profiler=profiler, suggester=suggester, validator=validator
    )

    try:
        loader = get_loader(datasource_config)
        df = loader.load(datasource_config)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise typer.Exit(code=1)

    if df.empty:
        logger.error("DataFrame is empty. Pipeline stopped.")
        raise typer.Exit(code=1)

    columns_config = config.get("columns_to_validate", [])

    if not columns_config or "__ALL__" in columns_config:
        columns_to_run = df.columns.tolist()
    else:
        columns_to_run = columns_config

    logger.info(f"Target columns for validation: {columns_to_run}")

    results = {}
    for col in columns_to_run:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found. Skipping.")
            continue
        logger.info(f"--- Processing column: {col} ---")

        current_col_relationships = []
        if all_relationships:
            for rel in all_relationships:
                if rel.get("from_column") == col:
                    current_col_relationships.append(rel)

        if current_col_relationships:
            logger.info(
                f"Found {len(current_col_relationships)} relationships for column '{col}'."
            )

        col_result = pipeline.run(
            df=df,
            column=col,
            open_docs=open_docs,
            relationships=current_col_relationships,
        )
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
    """
    Generates documentation for the specified columns in the config.yml file.
    """
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
        raise ValueError(f"Unknown profiler: {config['components']['profiler']}")

    doc_suggester = OllamaDocSuggester()

    datasource_config = config["datasource"]
    try:
        loader = get_loader(datasource_config)
        df = loader.load(datasource_config)
    except Exception as e:
        logger.error(f"Failed to initialize loader or load data: {e}")
        raise typer.Exit(code=1)

    if df.empty:
        logger.error("DataFrame is empty, pipeline stopped.")
        raise typer.Exit(code=1)

    columns_config = config.get("columns_to_validate", [])

    if not columns_config or "__ALL__" in columns_config:
        columns = df.columns.tolist()
    else:
        columns = columns_config

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
