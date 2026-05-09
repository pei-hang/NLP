from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from personal_assistant.memory import LongTermMemoryStore
from personal_assistant.tools import TOOLS, calculate_expression


SYSTEM_PROMPT = """你是一个个人对话式智能助手 Agent。

能力要求：
1. 自动判断是否需要调用工具：calculator、time_tool、web_search。
2. 使用短期对话记忆理解“刚才”“上一轮”“那个结果”等连续上下文。
3. 使用长期记忆进行个性化回答，特别是用户偏好、习惯和关键事实。
4. 回答必须使用中文，简洁、直接、可执行。

当前可用长期记忆：
{long_term_memory}

如果用户表达了偏好、习惯或长期事实，系统已在进入本轮推理前尝试写入长期记忆；你需要自然确认。
如果用户要求实时日期、当前时间或倒计时，必须调用 time_tool。
如果用户要求数学计算，必须调用 calculator。
如果用户询问实时新闻、最新信息、外部事实，必须调用 web_search。
"""


@dataclass
class AssistantResult:
    answer: str
    react_trace: str
    remembered: list[str] = field(default_factory=list)


class PersonalAssistantAgent:
    def __init__(
        self,
        memory_path: str = "data/long_term_memory.json",
        show_react: bool = True,
        mock: bool = False,
    ) -> None:
        load_dotenv()
        self.long_term_memory = LongTermMemoryStore(memory_path)
        self.chat_history: list[BaseMessage] = []
        self.show_react = show_react
        self.mock = mock
        self.executor: AgentExecutor | None = None
        if not mock:
            self.executor = self._build_executor()

    def _build_executor(self) -> AgentExecutor:
        api_key = os.getenv("ZHIPUAI_API_KEY") or os.getenv("ZHIPU_API_KEY")
        if not api_key:
            raise RuntimeError(
                "未找到 ZHIPUAI_API_KEY。请复制 .env.example 为 .env 并填写智谱 API Key，"
                "或使用 --mock 运行本地演示。"
            )

        llm = ChatOpenAI(
            model=os.getenv("ZHIPUAI_MODEL", "glm-4-flash"),
            api_key=api_key,
            base_url=os.getenv("ZHIPUAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/"),
            temperature=float(os.getenv("ASSISTANT_TEMPERATURE", "0.2")),
            timeout=float(os.getenv("ASSISTANT_TIMEOUT", "30")),
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )
        agent = create_tool_calling_agent(llm=llm, tools=TOOLS, prompt=prompt)
        return AgentExecutor(
            agent=agent,
            tools=TOOLS,
            verbose=False,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
        )

    def ask(self, user_input: str) -> AssistantResult:
        remembered = self.long_term_memory.update_from_user_message(user_input)
        memory_text = self.long_term_memory.format_for_prompt(user_input)

        if remembered and _is_memory_only_message(user_input):
            result = AssistantResult(
                answer="已记住这些偏好，后续回答和推荐会优先参考。",
                react_trace="\n".join(
                    [
                        "思考：用户表达了可复用的偏好或习惯，需要写入长期记忆，当前无需调用工具或等待大模型。",
                        "行动：无工具调用",
                        "观察：提取用户偏好并写入长期记忆，同时更新短期对话记忆。",
                        "回答：已记住这些偏好，后续回答和推荐会优先参考。",
                    ]
                ),
                remembered=remembered,
            )
        elif self.mock:
            result = self._ask_mock(user_input, remembered, memory_text)
        else:
            assert self.executor is not None
            payload = self.executor.invoke(
                {
                    "input": user_input,
                    "chat_history": self.chat_history,
                    "long_term_memory": memory_text,
                }
            )
            answer = str(payload["output"])
            result = AssistantResult(
                answer=answer,
                react_trace=self._format_react_trace(user_input, payload.get("intermediate_steps", []), answer, memory_text),
                remembered=remembered,
            )

        self.chat_history.append(HumanMessage(content=user_input))
        self.chat_history.append(AIMessage(content=result.answer))
        return result

    def _ask_mock(self, user_input: str, remembered: list[str], memory_text: str) -> AssistantResult:
        expression = _extract_math_expression(user_input)
        previous_number = _extract_last_number(self.chat_history)
        action = "无工具调用"
        observation = "已更新短期对话记忆。"

        if ("今天" in user_input or "几号" in user_input or "时间" in user_input) and "推荐" in user_input:
            from personal_assistant.tools import time_tool

            date_text = time_tool.invoke({"query": "date"})
            action = "调用 time_tool(date)，并读取长期记忆"
            observation = f"{date_text}；长期记忆：{memory_text}"
            answer = _drink_recommendation(memory_text, prefix=f"{date_text} ")
        elif remembered:
            observation = "提取用户偏好并写入长期记忆，同时更新短期对话记忆。"
            answer = "已记住这些偏好，后续推荐会优先参考。"
        elif expression:
            if "刚才" in user_input and previous_number is not None:
                expression = re.sub(r"刚才.*?结果|那个结果|上一轮.*?结果", str(previous_number), expression)
                if not re.search(r"\d", expression):
                    expression = f"{previous_number} * 2"
            value = calculate_expression(expression)
            action = f"调用 calculator({expression})"
            observation = f"得到结果 {value}，同步更新短期对话记忆。"
            answer = f"计算结果是 {value}。"
        elif "推荐" in user_input and ("饮品" in user_input or "喝" in user_input):
            action = "读取长期记忆"
            observation = f"检索到：{memory_text}"
            answer = _drink_recommendation(memory_text)
        elif "写" in user_input and "信" in user_input:
            action = "读取短期对话记忆和长期记忆"
            observation = f"长期记忆：{memory_text}"
            answer = (
                "给你一版简短信件：\n\n"
                "亲爱的你：\n"
                "有些话我想认真告诉你。遇见你以后，很多平常的时刻都变得特别起来。"
                "我喜欢和你相处时的自然，也喜欢想到你时心里安静又明亮的感觉。"
                "不管这封信会把我们带向哪里，我都想真诚地让你知道：你对我很重要。\n\n"
                "愿你今天也被温柔对待。"
            )
        else:
            answer = "我已收到。你可以继续提问，我会结合上下文和长期记忆回答。"

        trace = "\n".join(
            [
                f"思考：{_infer_thought(user_input, action, bool(remembered))}",
                f"行动：{action}",
                f"观察：{observation}",
                f"回答：{answer}",
            ]
        )
        return AssistantResult(answer=answer, react_trace=trace, remembered=remembered)

    def _format_react_trace(
        self,
        user_input: str,
        intermediate_steps: list[tuple[Any, Any]],
        answer: str,
        memory_text: str,
    ) -> str:
        thought = _infer_thought(user_input, "工具调用" if intermediate_steps else "无工具调用", False)
        lines = [f"思考：{thought}"]
        if intermediate_steps:
            for action, observation in intermediate_steps:
                tool_name = getattr(action, "tool", "unknown_tool")
                tool_input = getattr(action, "tool_input", "")
                lines.append(f"行动：调用 {tool_name}({tool_input})")
                lines.append(f"观察：{observation}")
        else:
            lines.append("行动：无工具调用")
            lines.append(f"观察：已读取长期记忆：{memory_text}")
        lines.append(f"回答：{answer}")
        return "\n".join(lines)


def _extract_math_expression(text: str) -> str | None:
    if "乘以" in text and "刚才" in text:
        tail_number = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
        return f"刚才结果 * {tail_number[-1]}" if tail_number else "刚才结果 * 2"
    candidate = text
    replacements = {
        "加": "+",
        "减": "-",
        "乘以": "*",
        "乘": "*",
        "除以": "/",
        "除": "/",
    }
    for old, new in replacements.items():
        candidate = candidate.replace(old, new)
    matches = re.findall(r"[-+*/().\d\s×÷xX^]+", candidate)
    expressions = [item.strip() for item in matches if re.search(r"\d", item) and re.search(r"[-+*/×÷xX^]", item)]
    return max(expressions, key=len) if expressions else None


def _is_memory_only_message(text: str) -> bool:
    if _extract_math_expression(text):
        return False
    request_markers = (
        "今天",
        "几号",
        "时间",
        "倒计时",
        "搜索",
        "联网",
        "最新",
        "实时",
        "新闻",
        "查一下",
        "查询",
        "一款",
        "适合",
        "写",
        "信",
        "生成",
        "帮我",
        "给我",
        "请",
        "算",
        "计算",
        "？",
        "?",
    )
    return not any(marker in text for marker in request_markers)


def _extract_last_number(history: list[BaseMessage]) -> str | None:
    for message in reversed(history):
        if isinstance(message, AIMessage):
            numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", str(message.content))
            if numbers:
                return numbers[-1]
    return None


def _drink_recommendation(memory_text: str, prefix: str = "") -> str:
    if "咖啡" in memory_text and ("冷饮" in memory_text or "冰" in memory_text):
        return f"{prefix}根据你的偏好，推荐冷萃冰咖啡，清爽、低甜，也避开了茶类。"
    if "咖啡" in memory_text:
        return f"{prefix}根据你的偏好，推荐冰美式或拿铁；如果想清爽一点，优先选冰美式。"
    return f"{prefix}推荐一杯低糖气泡水或冰咖啡；你告诉我更多口味后，我会记住并持续优化推荐。"


def _infer_thought(user_input: str, action: str, remembered: bool) -> str:
    if "calculator" in action or "计算" in user_input or "乘" in user_input:
        return "用户需要数学计算，需要结合短期上下文判断表达式并调用计算器。"
    if "time_tool" in action or "今天" in user_input or "几号" in user_input:
        return "用户需要日期时间信息，并要求结合长期偏好完成个性化回答。"
    if "推荐" in user_input:
        return "用户需要个性化推荐，应先读取长期记忆中的偏好。"
    if remembered:
        return "用户表达了可复用的偏好或习惯，需要写入长期记忆。"
    return "当前问题可直接回答，同时保留短期对话上下文。"
