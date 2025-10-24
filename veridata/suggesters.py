import logging
import json
import re
from abc import ABC, abstractmethod
import os


from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

RULE_SUGGESTER_PROMPT = """
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

DOC_SUGGESTER_PROMPT = """
            You are a Data Catalog Assistant. Your job is to write a concise, 
            one-sentence markdown description for a database column based on its profile.
            
            - Only output the description text. 
            - Do NOT include the column name (e.g., "This column is...")
            - Do NOT use markdown formatting (e.g., bold, italics).
            
            Data Profile:
            {profile_text}
            
            Example 1 Profile: {{ "column_name": "age", "min": 18, "max": 65, ... }}
            Example 1 Output: Represents the user's age, typically between 18 and 65.
            
            Example 2 Profile: {{ "column_name": "email_addr", "null_percentage": 0.1, ... }}
            Example 2 Output: The user's email address (can be null).

            Your turn.
            Data Profile:
            {profile_text}
            
            Output:
            """


def parse_llm_rules(raw_text: str):
    """
    Parses the raw text output from an LLM to extract a JSON array of rules.

    Args:
        raw_text (str): The raw text output from the LLM.

    Returns:
        list: A list of rule dictionaries.
    """
    array_match = re.search(r"\[[\s\S]*\]", raw_text)
    if array_match:
        try:
            return json.loads(array_match.group(0))
        except Exception:
            pass

    obj_match = re.search(r"\{[\s\S]*\}", raw_text)
    if obj_match:
        try:
            obj = json.loads(obj_match.group(0))
            return [obj]
        except Exception:
            pass

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
        print("JSON parsing failed completely, fallback to empty list.")
        print("Raw text:\n", raw_text)
        return []


class BaseSuggester(ABC):
    """
    Abstract base class for suggesters.
    """

    @abstractmethod
    def suggest(self, profile: dict) -> str:
        """
        Suggests rules or documentation based on a data profile.

        Args:
            profile (dict): The data profile.

        Returns:
            str: The suggested rules or documentation as a string.
        """
        pass


class OllamaRuleSuggester(BaseSuggester):
    """
    A rule suggester that uses an Ollama model to suggest Great Expectations rules.
    """

    def __init__(self, model: str = "gemma3:1b"):
        """
        Initializes the OllamaRuleSuggester.

        Args:
            model (str, optional): The name of the Ollama model to use. Defaults to "gemma3:1b".
        """
        logger.info(f"Initializing OllamaRuleSuggester with model '{model}'...")
        self.llm = ChatOllama(model=model, format="json")
        self.prompt_template = ChatPromptTemplate.from_template(
            RULE_SUGGESTER_PROMPT
        )
        self.parser = StrOutputParser()
        self.chain = self.prompt_template | self.llm | self.parser

    def suggest(self, profile: dict) -> str:
        """
        Suggests Great Expectations rules based on a data profile.

        Args:
            profile (dict): The data profile.

        Returns:
            str: The suggested rules as a JSON string.
        """
        logger.info(
            f"Suggesting rules via LLM for '{profile['column_name']}'..."
        )
        profile_text = json.dumps(profile, indent=2)
        response = self.chain.invoke({"profile_text": profile_text})
        logger.info(f"LLM Raw Output: {response}")
        rules_list = parse_llm_rules(response)
        return json.dumps(rules_list, indent=2)


class OllamaDocSuggester(BaseSuggester):
    """
    A documentation suggester that uses an Ollama model to suggest column descriptions.
    """

    def __init__(self, model: str = "gemma3:1b"):
        """
        Initializes the OllamaDocSuggester.

        Args:
            model (str, optional): The name of the Ollama model to use. Defaults to "gemma3:1b".
        """
        logger.info(f"Initializing OllamaDocSuggester with model '{model}'...")

        self.llm = ChatOllama(model=model)

        self.prompt_template = ChatPromptTemplate.from_template(
            DOC_SUGGESTER_PROMPT
        )
        self.parser = StrOutputParser()
        self.chain = self.prompt_template | self.llm | self.parser

    def suggest(self, profile: dict) -> str:
        """
        Suggests a column description based on a data profile.

        Args:
            profile (dict): The data profile.

        Returns:
            str: The suggested column description.
        """
        logger.info(
            f"Suggesting documentation via LLM for '{profile['column_name']}'..."
        )
        profile_text = json.dumps(profile, indent=2)
        response = self.chain.invoke({"profile_text": profile_text})
        logger.info(f"LLM Raw Output: {response}")
        return response


class OpenAIRuleSuggester(BaseSuggester):
    """
    A rule suggester that uses an OpenAI model to suggest Great Expectations rules.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initializes the OpenAIRuleSuggester.

        Args:
            model (str, optional): The name of the OpenAI model to use. Defaults to "gpt-4o-mini".
        """
        logger.info(f"Initializing OpenAIRuleSuggester with model '{model}'...")

        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set.")

        self.llm = ChatOpenAI(
            model=model,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        self.prompt_template = ChatPromptTemplate.from_template(
            RULE_SUGGESTER_PROMPT
        )
        self.parser = StrOutputParser()
        self.chain = self.prompt_template | self.llm | self.parser

    def suggest(self, profile: dict) -> str:
        """
        Suggests Great Expectations rules based on a data profile.

        Args:
            profile (dict): The data profile.

        Returns:
            str: The suggested rules as a JSON string.
        """
        logger.info(
            f"Suggesting rules via LLM for '{profile['column_name']}'..."
        )
        profile_text = json.dumps(profile, indent=2)

        response = self.chain.invoke({"profile_text": profile_text})
        logger.info(f"LLM Raw Output: {response}")

        rules_list = parse_llm_rules(response)
        return json.dumps(rules_list, indent=2)
