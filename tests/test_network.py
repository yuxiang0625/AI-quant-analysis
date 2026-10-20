import requests

url = "https://query1.finance.yahoo.com"

try:
    response = requests.get(url, timeout=15)

    print("HTTP 状态码：", response.status_code)
    print("请求网址：", response.url)

except requests.exceptions.RequestException as e:
    print("请求失败：", e)