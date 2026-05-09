from __future__ import annotations

import ast
import operator
import re
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from langchain_core.tools import tool


_ALLOWED_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def calculate_expression(expression: str) -> str:
    """Safely evaluate a numeric expression without using eval."""
    normalized = (
        expression.strip()
        .replace("×", "*")
        .replace("x", "*")
        .replace("X", "*")
        .replace("÷", "/")
        .replace("^", "**")
    )
    tree = ast.parse(normalized, mode="eval")
    result = _eval_numeric_node(tree.body)
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


def _eval_numeric_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BIN_OPS:
        left = _eval_numeric_node(node.left)
        right = _eval_numeric_node(node.right)
        return _ALLOWED_BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY_OPS:
        return _ALLOWED_UNARY_OPS[type(node.op)](_eval_numeric_node(node.operand))
    raise ValueError("只支持数字、括号和常见四则运算符。")


@tool
def calculator(expression: str) -> str:
    """计算数学表达式。输入示例：256 * 1024、(12 + 8) / 4。"""
    try:
        return calculate_expression(expression)
    except Exception as exc:
        return f"计算失败：{exc}"


@tool
def time_tool(query: str = "now") -> str:
    """查询当前日期时间，或计算到某个日期时间的倒计时。输入示例：now、date、countdown 2026-06-01 09:00。"""
    tz = ZoneInfo("Asia/Shanghai")
    now = datetime.now(tz)
    lower_query = query.strip().lower()

    if "countdown" in lower_query or "倒计时" in lower_query:
        target_text = re.sub(r"countdown|倒计时|到", "", query, flags=re.IGNORECASE).strip()
        target = _parse_target_datetime(target_text, tz)
        if target is None:
            return "倒计时解析失败，请使用格式：countdown 2026-06-01 09:00"
        delta = target - now
        total_seconds = int(delta.total_seconds())
        if total_seconds < 0:
            return f"{target.strftime('%Y-%m-%d %H:%M:%S')} 已经过了。"
        days, remainder = divmod(total_seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"距离 {target.strftime('%Y-%m-%d %H:%M:%S')} 还有 {days} 天 {hours} 小时 {minutes} 分 {seconds} 秒。"

    if "date" in lower_query or "日期" in lower_query or "几号" in lower_query:
        weekdays = ["一", "二", "三", "四", "五", "六", "日"]
        return now.strftime(f"今天是 %Y 年 %m 月 %d 日，星期{weekdays[now.weekday()]}。")
    return now.strftime("当前时间是 %Y-%m-%d %H:%M:%S，时区 Asia/Shanghai。")


def _parse_target_datetime(text: str, tz: ZoneInfo) -> datetime | None:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            parsed = datetime.strptime(text, fmt)
            return parsed.replace(tzinfo=tz)
        except ValueError:
            continue
    return None


@tool
def web_search(query: str) -> str:
    """联网搜索实时信息。输入应是完整搜索问题。网络或依赖不可用时返回模拟搜索提示。"""
    try:
        from langchain_community.tools import DuckDuckGoSearchRun

        search = DuckDuckGoSearchRun()
        return search.run(query)
    except Exception as exc:
        return (
            "模拟搜索结果：当前环境未能完成联网搜索，"
            f"查询词为「{query}」。请安装 ddgs 并确认网络可用。错误：{exc}"
        )


TOOLS: list[Any] = [calculator, time_tool, web_search]
