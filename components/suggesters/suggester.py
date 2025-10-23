import logging
from typing import Dict, Any
from abc import ABC, abstractmethod

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)


class BaseRuleSuggester(ABC):
    @abstractmethod
    def suggest_rules(self, column_profile: Dict[str, Any], column_name: str) -> str:
        pass


class OllamaRuleSuggester(BaseRuleSuggester):
    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url

        try:
            llm = OllamaLLM(model=self.model_name, base_url=self.base_url)
            prompt_template = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "당신은 데이터 품질 규칙을 제안하는 전문가입니다. 주어진 컬럼 정보에 가장 적합한 규칙 2가지를 제안해주세요.",
                    ),
                    (
                        "human",
                        """
                데이터 컬럼 '{column_name}' 분석 결과:
                - 데이터 타입: {dtype}
                - Null 값 비율: {null_percentage:.2f}%
                - 고유값 비율: {unique_percentage:.2f}%
                - 샘플 데이터: {sample_data}

                이 정보를 바탕으로, 이 컬럼의 의미를 추론하고, 적용할 수 있는 데이터 품질 규칙 2가지를 제안해주세요.
                각 규칙에 대해 다음 정보를 포함해주세요:
                1. 규칙 이름 (예: check_not_null, check_unique, check_email_format, check_value_range)
                2. 규칙 설명 (예: Null 값이 없어야 합니다.)
                3. (선택) 규칙 적용 시 필요한 파라미터 (예: 범위 검사의 경우 min_value, max_value)
                
                Do not use Markdown formatting.
                """,
                    ),
                ]
            )
            output_parser = StrOutputParser()
            self.chain = prompt_template | llm | output_parser
            logger.info(
                f"OllamaRuleSuggester 초기화 완료 (모델: {self.model_name}, URL: {self.base_url}"
            )
        except Exception as e:
            logger.error(f"OllamaRuleSuggester 초기화 중 오류 발생: {e}", exc_info=True)
            self.chain = None

    def suggest_rules(self, column_profile: Dict[str, Any], column_name: str) -> str:
        if self.chain is None:
            return "오류: OllamaRuleSuggester가 초기화되지 않았습니다."

        logger.info(f"'{column_name}' 컬럼에 대한 LLM 기반 규칙 제안을 시작합니다...")
        try:
            # 프로파일링 결과에서 필요한 정보 추출
            dtype = column_profile.get("dtype", "unknown")
            null_percentage = column_profile.get("null_percentage", 0.0)
            unique_percentage = column_profile.get("unique_percentage", 0.0)
            sample_data = column_profile.get("sample_data", [])

            # LangChain 체인 실행
            suggestion = self.chain.invoke(
                {
                    "column_name": column_name,
                    "dtype": dtype,
                    "null_percentage": null_percentage,
                    "unique_percentage": unique_percentage,
                    "sample_data": sample_data,
                }
            )
            logger.info(f"'{column_name}' 컬럼에 대한 LLM 기반 규칙 제안 완료.")
            return suggestion
        except Exception as e:
            logger.error(f"LangChain/Ollama 실행 중 에러 발생: {e}", exc_info=True)
            return "오류: LangChain/Ollama 실행 중 에러가 발생했습니다."
