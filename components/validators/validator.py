import logging
import pandas as pd
from typing import Dict, Any, List
from abc import ABC, abstractmethod
import great_expectations as gx
from great_expectations.core.batch import (
    RuntimeBatchRequest,
)  # ✨ RuntimeBatchRequest 임포트

logger = logging.getLogger(__name__)


# --- BaseValidator (동일) ---
class BaseValidator(ABC):
    @abstractmethod
    def validate(self, df: pd.DataFrame, rules: List[Dict[str, Any]]) -> Dict[str, Any]:
        pass


# --- GreatExpectationsValidator (최신 API 적용) ---
class GreatExpectationsValidator(BaseValidator):
    """
    Great Expectations 라이브러리(v3 API, 임시 컨텍스트)를 사용하여
    데이터 품질 규칙을 검증하는 클래스입니다.
    """

    def validate(self, df: pd.DataFrame, rules: List[Dict[str, Any]]) -> Dict[str, Any]:
        logger.info(f"{len(rules)}개의 규칙으로 데이터 검증을 시작합니다...")

        # ✨ 1. 임시(Ephemeral) 데이터 컨텍스트를 얻습니다.
        context = gx.get_context(mode="ephemeral")

        # ✨ 2. DataFrame을 사용하여 '런타임 배치 요청'을 생성합니다.
        #    이것이 GX v3에서 메모리 내 DataFrame을 처리하는 핵심 방법입니다.
        batch_request = RuntimeBatchRequest(
            datasource_name="my_pandas_datasource",  # 임시 데이터소스 이름
            data_asset_name="my_dataframe_asset",  # 임시 데이터 에셋 이름
            runtime_data=df,  # 실제 DataFrame 전달
            batch_identifier="default_identifier",  # 배치 식별자
        )

        # ✨ 3. 배치 요청을 사용하여 Validator를 가져옵니다.
        validator = context.get_validator(batch_request=batch_request)

        # ✨ 4. Validator 객체에 Expectation 적용 (이전 로직과 거의 동일)
        #    이제 getattr을 다시 안전하게 사용할 수 있습니다.
        expectation_suite_name = "temp_suite"  # 임시 이름
        validator.expectation_suite_name = expectation_suite_name

        for rule in rules:
            column_name = rule.get("column")
            rule_name = rule.get("rule_name")
            params = rule.get("params", {})

            if not column_name or not rule_name:
                logger.warning(f"잘못된 규칙 형식: {rule}. 건너<0xEB><0x9B><0x8D>니다.")
                continue

            # 간단한 규칙 이름 -> GX 메소드 이름 변환
            # (향후 더 많은 규칙을 지원하려면 이 부분을 확장해야 함)
            if rule_name == "check_not_null":
                expectation_name = "expect_column_values_to_not_be_null"
            elif rule_name == "check_email_format":
                expectation_name = "expect_column_values_to_match_regex"
                params["regex"] = r"[^@]+@[^@]+\.[^@]+"  # 이메일 정규식 추가
            else:
                logger.warning(f"지원하지 않는 규칙 이름: '{rule_name}'")
                continue  # 알 수 없는 규칙은 건너<0xEB><0x9B><0x8D>니다.

            try:
                expectation_method = getattr(validator, expectation_name)
                logger.debug(
                    f"Applying rule '{expectation_name}' to column '{column_name}' with params: {params}"
                )
                # 파라미터와 함께 expectation 메소드 호출
                expectation_method(column=column_name, **params)
            except AttributeError:
                # getattr 실패 시 (메소드가 없을 경우)
                logger.warning(
                    f"Great Expectations에 해당 규칙 메소드가 없습니다: '{expectation_name}'"
                )
            except Exception as e:
                logger.error(
                    f"규칙 '{expectation_name}' 적용 중 오류 발생 (Column: {column_name}): {e}",
                    exc_info=True,
                )

        logger.info("모든 규칙 적용 완료. 최종 검증을 실행합니다...")
        # ✨ 5. Validator를 사용하여 검증 실행 (이전과 동일)
        validation_result = validator.validate()

        return validation_result.to_json_dict()  # 결과를 JSON 호환 딕셔너리로 반환
