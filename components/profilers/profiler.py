import pandas as pd
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class PandasProfiler:
    def profile(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        logger.info(f"DataFrame 프로파일링 시작(Shape: df.shape)")
        profile_results = {}

        if df.empty:
            logger.warning("DataFrame이 비어 있습니다.")
            return profile_results

        total_rows = len(df)
        null_counts = df.isnull().sum()
        unique_counts = df.nunique().sum()

        for col_name in df.columns:
            column_data = df[col_name]
            col_profile: Dict[str, Any] = {}

            col_profile["dtype"] = str(column_data.dtype)
            col_profile["total_count"] = total_rows
            col_profile["non_null_count"] = int(
                total_rows - null_counts[col_name]
            )
            col_profile["null_percentage"] = float(
                null_counts[col_name] / total_rows * 100
            )

            if col_profile["non_null_count"] > 0:
                unique_values = column_data.nunique()
                col_profile["unique_count"] = int(unique_values)
                col_profile["unique_percentage"] = float(
                    unique_values / col_profile["non_null_count"] * 100
                )
            else:
                col_profile["unique_count"] = 0
                col_profile["unique_percentage"] = 0.0

            try:
                # 숫자형 컬럼 통계
                if pd.api.types.is_numeric_dtype(column_data.dtype):
                    stats = column_data.describe().to_dict()
                    # NaN 값을 None으로 변환 (JSON 직렬화 가능하도록)
                    col_profile["stats"] = {
                        k: (None if pd.isna(v) else v) for k, v in stats.items()
                    }
                # 문자열/범주형 컬럼 통계
                elif pd.api.types.is_object_dtype(
                    column_data.dtype
                ) or pd.api.types.is_categorical_dtype(column_data.dtype):
                    stats = column_data.describe().to_dict()
                    col_profile["stats"] = {
                        k: (None if pd.isna(v) else v) for k, v in stats.items()
                    }
                # 날짜/시간 컬럼 (기본 통계만 제공)
                elif pd.api.types.is_datetime64_any_dtype(column_data.dtype):
                    stats = column_data.describe(
                        datetime_is_numeric=True
                    ).to_dict()
                    col_profile["stats"] = {
                        k: (None if pd.isna(v) else v) for k, v in stats.items()
                    }

            except Exception as e:
                logger.warning(f"'{col_name}' 컬럼 통계 계산 중 오류 발생: {e}")
                col_profile["stats"] = {}

            # 4. 샘플 데이터 (LLM 프롬프트용)
            if col_profile["non_null_count"] > 0:
                sample_size = min(5, col_profile["non_null_count"])
                col_profile["sample_data"] = (
                    column_data.dropna()
                    .sample(n=sample_size, random_state=42)
                    .tolist()
                )
            else:
                col_profile["sample_data"] = []

            profile_results[col_name] = col_profile

        logger.info("DataFrame 프로파일링 완료.")
        return profile_results
