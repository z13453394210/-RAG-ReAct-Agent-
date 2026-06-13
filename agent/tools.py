"""
rag-knowledge-agent — 工具定义

每个工具都是一个 LangChain StructuredTool，Agent 可以调用它们。
"""

from typing import Optional

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from agent.rag import search_knowledge_base, format_retrieved_docs


# ===================================================================
# 1. 知识库（RAG）工具
# ===================================================================
class KBInput(BaseModel):
    query: str = Field(description="要用中文提出的自然语言问题，用于搜索知识库。")


def _kb_search(query: str) -> str:
    """在已上传的文档中搜索相关信息。"""
    results = search_knowledge_base(query, k=5)
    if not results:
        return "知识库中没有找到相关信息。"
    return format_retrieved_docs(results)


kb_tool = StructuredTool.from_function(
    func=_kb_search,
    name="knowledge_base",
    description=(
        "搜索已上传的知识库文档，查找与用户问题相关的信息。"
        "当用户询问上传文档中的内容时使用此工具。"
    ),
    args_schema=KBInput,
)


# ===================================================================
# 2. 联网搜索工具（DuckDuckGo，无需 API Key）
# ===================================================================
class WebSearchInput(BaseModel):
    query: str = Field(description="搜索关键词。")


def _web_search(query: str) -> str:
    """搜索互联网获取实时信息。"""
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return "没有找到搜索结果。"
        lines: list[str] = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "无标题")
            snippet = r.get("body", "")
            link = r.get("href", "")
            lines.append(f"{i}. **{title}**\n   {snippet}\n   {link}")
        return "\n".join(lines)
    except ImportError:
        return "联网搜索不可用（未安装 duckduckgo_search）。"
    except Exception as e:
        return f"搜索失败: {e}"


web_search_tool = StructuredTool.from_function(
    func=_web_search,
    name="web_search",
    description=(
        "搜索互联网获取实时或最新信息。"
        "当答案涉及时效性、不在知识库中、"
        "或者知识库建立后可能已变化时使用。"
    ),
    args_schema=WebSearchInput,
)


# ===================================================================
# 3. 天气查询工具（Open-Meteo，免费，无需 API Key）
# ===================================================================
class WeatherInput(BaseModel):
    location: str = Field(description="城市名称，例如 '北京'、'上海'、'New York'。")
    days: int = Field(default=1, description="预报天数（1-7）。")


def _get_coordinates(city: str) -> Optional[tuple[float, float]]:
    """通过 Open-Meteo 地理编码 API 获取城市坐标。"""
    import httpx

    resp = httpx.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1, "language": "zh", "format": "json"},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    if not data.get("results"):
        return None
    r = data["results"][0]
    return r["latitude"], r["longitude"]


def _weather_forecast(location: str, days: int = 1) -> str:
    """获取指定城市的天气预报。"""
    try:
        coords = _get_coordinates(location)
        if coords is None:
            return f"找不到城市 '{location}' 的坐标信息。"
        lat, lon = coords

        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": ["temperature_2m_max", "temperature_2m_min",
                      "precipitation_sum", "weather_code"],
            "timezone": "auto",
            "forecast_days": min(max(days, 1), 7),
        }
        resp = httpx.get(
            "https://api.open-meteo.com/v1/forecast",
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        daily = data.get("daily", {})
        times = daily.get("time", [])
        t_max = daily.get("temperature_2m_max", [])
        t_min = daily.get("temperature_2m_min", [])
        precip = daily.get("precipitation_sum", [])
        codes = daily.get("weather_code", [])

        # WMO 天气代码转中文描述
        wmo_codes = {
            0: "晴天", 1: "大部晴朗", 2: "多云", 3: "阴天",
            45: "有雾", 48: "雾凇",
            51: "小毛毛雨", 53: "中毛毛雨", 55: "大毛毛雨",
            61: "小雨", 63: "中雨", 65: "大雨",
            71: "小雪", 73: "中雪", 75: "大雪",
            80: "小阵雨", 81: "中阵雨", 82: "大暴雨",
            95: "雷暴", 96: "雷暴伴小冰雹", 99: "雷暴伴大冰雹",
        }

        lines = [f"**{location}**（{lat:.2f}°N, {lon:.2f}°E）天气预报：\n"]
        for i in range(len(times)):
            date = times[i]
            hi = t_max[i] if i < len(t_max) else "N/A"
            lo = t_min[i] if i < len(t_min) else "N/A"
            pr = precip[i] if i < len(precip) else 0
            wmo = codes[i] if i < len(codes) else 0
            desc = wmo_codes.get(wmo, f"代码 {wmo}")
            lines.append(
                f"  - **{date}**：{desc}，"
                f"🌡 {lo}°C – {hi}°C，💧 {pr}mm"
            )
        return "\n".join(lines)

    except ImportError:
        return "天气查询不可用（未安装 httpx）。"
    except Exception as e:
        return f"天气查询失败: {e}"


weather_tool = StructuredTool.from_function(
    func=_weather_forecast,
    name="weather",
    description=(
        "查询任意城市的当前天气或预报。"
        "输入城市名称，可选填预报天数（默认 1 天）。"
    ),
    args_schema=WeatherInput,
)


# ===================================================================
# 4. 计算器工具
# ===================================================================
class CalculatorInput(BaseModel):
    expression: str = Field(description="要计算的数学表达式，例如 '2 + 2' 或 'sqrt(144)'。")


def _calculate(expression: str) -> str:
    """安全地计算数学表达式。"""
    allowed_names = {
        k: v for k, v in __import__("math").__dict__.items()
        if not k.startswith("_")
    }
    allowed_names.update({"abs": abs, "round": round, "min": min, "max": max})
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"


calculator_tool = StructuredTool.from_function(
    func=_calculate,
    name="calculator",
    description="计算数学表达式。支持 +、-、*、/、sqrt、sin、cos 等。",
    args_schema=CalculatorInput,
)


# ===================================================================
# 工具注册表
# ===================================================================
ALL_TOOLS: list[BaseTool] = [
    kb_tool,
    web_search_tool,
    weather_tool,
    calculator_tool,
]


def get_tools(names: Optional[list[str]] = None) -> list[BaseTool]:
    """按名称返回工具子集，如果 names 为 None 则返回全部工具。"""
    if names is None:
        return ALL_TOOLS
    name_set = set(names)
    return [t for t in ALL_TOOLS if t.name in name_set]