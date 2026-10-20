import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

FRED_API_KEY = os.getenv("FRED_API_KEY")
FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"


# FRED 公开经济数据
FRED_SERIES = {
    "DGS10": {
        "name": "美国10年期国债收益率",
        "unit": "%",
        "frequency": "日度",
    },
    "DFII10": {
        "name": "美国10年期实际利率",
        "unit": "%",
        "frequency": "日度",
    },
    "CPIAUCSL": {
        "name": "美国消费者价格指数",
        "unit": "指数",
        "frequency": "月度",
    },
    "FEDFUNDS": {
        "name": "美国联邦基金有效利率",
        "unit": "%",
        "frequency": "月度",
    },
}


def fetch_fred_series(series_id: str) -> pd.DataFrame:
    """通过 FRED 官方 API 获取宏观经济数据。"""

    if series_id not in FRED_SERIES:
        raise ValueError(f"不支持的 FRED 指标：{series_id}")

    if not FRED_API_KEY:
        raise RuntimeError(
            "没有找到 FRED_API_KEY，请检查项目根目录下的 .env 文件。"
        )

    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "sort_order": "asc",
        "limit": 100000,
    }

    last_error = None

    for attempt in range(3):
        try:
            response = requests.get(
                FRED_API_URL,
                params=params,
                timeout=(10, 60),
            )
            response.raise_for_status()

            result = response.json()

            if "error_code" in result:
                raise RuntimeError(
                    f"FRED API 错误：{result.get('error_message')}"
                )

            observations = result.get("observations", [])

            data = pd.DataFrame(observations)

            if data.empty:
                raise RuntimeError(
                    f"{series_id} 没有返回观测数据。"
                )

            data = data.rename(
                columns={
                    "date": "observation_date",
                }
            )

            data["observation_date"] = pd.to_datetime(
                data["observation_date"],
                errors="coerce",
            )

            data["value"] = pd.to_numeric(
                data["value"],
                errors="coerce",
            )

            data = data.dropna(
                subset=["observation_date", "value"]
            )

            data["indicator"] = series_id
            data["name"] = FRED_SERIES[series_id]["name"]
            data["unit"] = FRED_SERIES[series_id]["unit"]
            data["frequency"] = FRED_SERIES[series_id]["frequency"]
            data["source"] = "FRED"

            return data[
                [
                    "observation_date",
                    "indicator",
                    "name",
                    "value",
                    "unit",
                    "frequency",
                    "source",
                ]
            ].sort_values(
                "observation_date"
            ).reset_index(drop=True)

        except Exception as exc:
            last_error = exc

            if attempt < 2:
                import time
                time.sleep(2 * (attempt + 1))

    raise RuntimeError(
        f"获取 {series_id} 失败，已重试 3 次：{last_error}"
    )


def fetch_all_macro_data() -> pd.DataFrame:
    """
    获取第一批宏观经济指标。

    如果某个指标获取失败，保留其他成功获取的数据。
    """

    results = []
    errors = []

    for series_id in FRED_SERIES:
        try:
            data = fetch_fred_series(series_id)

            if not data.empty:
                results.append(data)

            print(
                f"获取成功：{series_id}，"
                f"{len(data)} 条记录"
            )

        except Exception as exc:
            errors.append(
                f"{series_id}: {exc}"
            )

            print(
                f"获取失败：{series_id}，原因：{exc}"
            )

    if not results:
        raise RuntimeError(
            "所有宏观经济指标均获取失败。\n"
            + "\n".join(errors)
        )

    combined = pd.concat(
        results,
        ignore_index=True,
    )

    if errors:
        print("\n部分指标获取失败：")
        for error in errors:
            print("-", error)

    return combined


if __name__ == "__main__":
    macro_data = fetch_all_macro_data()

    print("\n宏观经济数据获取完成！")
    print(f"总记录数：{len(macro_data)}")

    print("\n各指标记录数量：")
    print(
        macro_data.groupby(
            ["indicator", "name"]
        ).size().to_string()
    )

    print("\n最近的五条数据：")
    print(
        macro_data.tail(5).to_string(
            index=False
        )
    )