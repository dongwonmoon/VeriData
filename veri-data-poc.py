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

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_llm_rules(raw_text: str):
    """
    gemma가 아무리 개판으로 JSON을 뱉어도,
    무조건 유효한 Python list로 복원하는 버전.
    """
    # 1️⃣ JSON 배열이 있는 경우 그대로
    array_match = re.search(r"\[[\s\S]*\]", raw_text)
    if array_match:
        try:
            return json.loads(array_match.group(0))
        except Exception:
            pass  # 다음 단계로

    # 2️⃣ JSON 객체가 있는 경우 → 리스트로 감싸기
    obj_match = re.search(r"\{[\s\S]*\}", raw_text)
    if obj_match:
        try:
            obj = json.loads(obj_match.group(0))
            return [obj]
        except Exception:
            pass

    # 3️⃣ JSON 파싱 완전 실패 → eval 유사 fallback (매우 관대한 파서)
    try:
        raw_text_fixed = (
            raw_text.strip()
            .replace("True", "true")
            .replace("False", "false")
            .replace("None", "null")
        )
        data = json.loads(raw_text_fixed)
        return data if isinstance(data, list) else [data]
    except Exception:
        # 4️⃣ 그래도 안되면 빈 리스트 반환 (파이프라인 끊기지 않게)
        print("⚠️ JSON parsing failed completely, fallback to empty list.")
        print("Raw text:\n", raw_text)
        return []


def load_data(csv_path: str) -> pd.DataFrame:
    logger.info(f"S1: Loading data from {csv_path}")
    try:
        return pd.read_csv(csv_path)
    except FileNotFoundError:
        logger.error(f"File not found: {csv_path}")
        return pd.DataFrame()


def profile_data(df: pd.DataFrame, column: str) -> dict:
    logger.info(f"S2: Profiling column '{column}'...")
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


def suggest_rules_json(profile: dict) -> str:
    logger.info(f"S3: Suggesting rules via LLM for '{profile['column_name']}'...")
    llm = ChatOllama(model="gemma3:1b", format="json")
    prompt_template = ChatPromptTemplate.from_template(
        """
        You are a data quality assistant that outputs ONLY valid JSON.
        Your output will be rejected if it contains ANY text outside the JSON array.
        
        INSTRUCTIONS (read carefully):
        - You MUST output a valid JSON **array** of Great Expectations rule objects.
        - DO NOT include any explanations, commentary, markdown code blocks, or text before/after the JSON.
        - The output must begin with '[' and end with ']'.
        - Each item in the array must have two keys:
            - "expectation_type" (string)
            - "kwargs" (dictionary of parameters for the expectation)
        - Every "kwargs" must contain at least a "column" key with the target column name.
        - Output must be parseable by json.loads() with no modification.
        - You must output a JSON *array* (use square brackets [ ]) even if there is only one rule.
        - Example of single rule output: [{{"expectation_type": "expect_column_values_to_not_be_null", "kwargs": {{"column": "age"}}}}]

        Example Output (this is the ONLY acceptable format):
        [
        {{
            "expectation_type": "expect_column_values_to_not_be_null",
            "kwargs": {{ "column": "user_email", "mostly": 0.95 }}
        }},
        {{
            "expectation_type": "expect_column_values_to_be_between",
            "kwargs": {{ "column": "age", "min_value": 0, "max_value": 120 }}
        }}
        ]

        Now, based on this data profile, generate 2-3 rules:
        {profile_text}
        """
    )
    parser = StrOutputParser()
    chain = prompt_template | llm | parser
    profile_text = json.dumps(profile, indent=2)
    response = chain.invoke({"profile_text": profile_text})
    logger.info(f"LLM Raw Output: {response}")
    rules_list = parse_llm_rules(response)
    return json.dumps(rules_list, indent=2)


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
    if not df.empty:
        profile = profile_data(df, "age")
        if profile:
            suggested_rules = suggest_rules_json(profile)
            print("--- LLM Suggested Rules ---")
            print(suggested_rules)

            validation_result = validate_data(df, "age", suggested_rules)
            print("\n--- Validation Results ---")
            print(json.dumps(validation_result, indent=2))
        else:
            logger.warning("Profile is empty, skipping rule suggestion.")
    else:
        logger.warning("DataFrame is empty, pipeline stopped.")
