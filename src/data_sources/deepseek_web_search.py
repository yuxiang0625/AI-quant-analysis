import os
import json
import requests
from pathlib import Path
from dotenv import load_dotenv


# =========================
# 1. 项目配置
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("DEEPSEEK_API_KEY")

SEARCH_URL = "https://api.deepseek.com/v1/tools/search"

TIMEOUT = 30


# =========================
# 2. 调用 DeepSeek 搜索
# =========================

def search_web(query, count=5):
    """
    调用 DeepSeek 搜索 API。
    返回 API 的原始 JSON，避免假设返回字段结构。
    """

    if not API_KEY:
        raise RuntimeError(
            "没有找到 DEEPSEEK_API_KEY，请检查项目根目录下的 .env 文件。"
        )

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "query": query,
        "count": count,
        "need_content": True,
    }

    response = requests.post(
        SEARCH_URL,
        headers=headers,
        json=payload,
        timeout=TIMEOUT,
    )

    if not response.ok:
        raise RuntimeError(
            f"DeepSeek 搜索请求失败。\n"
            f"HTTP 状态码：{response.status_code}\n"
            f"返回内容：{response.text[:1000]}"
        )

    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"API 返回的不是有效 JSON：{response.text[:500]}"
        ) from exc


# =========================
# 3. 搜索 COMEX 数据源
# =========================

def search_comex_sources():
    """
    分别搜索官方报告、历史数据和公开 API。
    """

    queries = [
        (
            "COMEX gold warehouse stocks daily report "
            "official CME registered eligible total"
        ),
        (
            "COMEX gold inventory historical data CSV "
            "download registered eligible"
        ),
        (
            "COMEX gold inventory API daily historical data "
            "public data source"
        ),
    ]

    all_results = []

    for index, query in enumerate(queries, start=1):
        print("\n" + "=" * 65)
        print(f"搜索任务 {index}/{len(queries)}")
        print(f"关键词：{query}")
        print("=" * 65)

        try:
            result = search_web(query)

            # 保存完整原始结果，便于检查实际返回结构
            all_results.append({
                "query": query,
                "response": result,
            })

            print(json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            ))

        except (requests.RequestException, RuntimeError) as exc:
            print(f"本次搜索失败：{exc}")

    return all_results


# =========================
# 4. 保存搜索记录
# =========================

def save_search_results(results):
    """
    保存 AI 搜索结果，不把搜索摘要当作行情数据。
    """

    output_dir = PROJECT_ROOT / "data" / "comex_search"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "search_results.json"

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n搜索记录已保存：")
    print(output_file)


# =========================
# 5. 主程序
# =========================

def main():
    print("=" * 65)
    print("Aurum Intelligence | COMEX AI 数据源发现")
    print("=" * 65)

    results = search_comex_sources()

    if not results:
        print("\n没有获取到搜索结果，请检查报错信息。")
        return

    save_search_results(results)

    print("\n" + "=" * 65)
    print("搜索任务完成。")
    print("注意：目前保存的是数据源搜索结果，不是库存行情。")
    print("下一步需要根据真实结果验证数据下载地址和字段。")
    print("=" * 65)


if __name__ == "__main__":
    main()