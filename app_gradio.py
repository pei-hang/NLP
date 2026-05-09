from __future__ import annotations

import os
import socket
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

from personal_assistant.agent import PersonalAssistantAgent


MEMORY_PATH = "data/long_term_memory.json"
AGENT: PersonalAssistantAgent | None = None
CURRENT_MODE = ""


CSS = """
:root {
  --agent-blue: #2563eb;
  --agent-ink: #111827;
  --agent-muted: #64748b;
}
.gradio-container {
  max-width: 1280px !important;
  margin: 0 auto !important;
  font-family: "PingFang SC", "Microsoft YaHei", system-ui, sans-serif !important;
}
#hero {
  background: linear-gradient(135deg, #0f172a 0%, #1e293b 52%, #0f766e 100%);
  border-radius: 18px;
  padding: 26px 30px;
  color: white;
  border: 1px solid rgba(255,255,255,0.12);
}
#hero h1 {
  font-size: 30px;
  line-height: 1.25;
  margin: 0 0 8px;
  letter-spacing: 0;
}
#hero p {
  color: #dbeafe;
  margin: 0;
  font-size: 15px;
}
.panel {
  border: 1px solid #dbe4ef !important;
  border-radius: 14px !important;
  background: #ffffff !important;
  box-shadow: 0 12px 34px rgba(15, 23, 42, 0.08);
}
.memory-card {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  padding: 12px 14px;
  margin: 8px 0;
}
.status-ok {
  color: #0f766e;
  font-weight: 700;
}
.status-warn {
  color: #b45309;
  font-weight: 700;
}
footer {display: none !important;}
"""


def _create_agent(mode: str) -> tuple[PersonalAssistantAgent, str, str]:
    load_dotenv()
    requested_mock = mode == "Mock 演示"
    if requested_mock:
        return PersonalAssistantAgent(memory_path=MEMORY_PATH, mock=True), "Mock 演示", "当前使用 Mock 演示模式。"

    has_key = bool(os.getenv("ZHIPUAI_API_KEY") or os.getenv("ZHIPU_API_KEY"))
    if not has_key:
        return (
            PersonalAssistantAgent(memory_path=MEMORY_PATH, mock=True),
            "Mock 演示",
            "未检测到智谱 API Key，已自动切换到 Mock 演示模式。",
        )

    try:
        return PersonalAssistantAgent(memory_path=MEMORY_PATH, mock=False), "智谱 API", "当前使用智谱 API 模式。"
    except RuntimeError as exc:
        return PersonalAssistantAgent(memory_path=MEMORY_PATH, mock=True), "Mock 演示", f"{exc} 已切换到 Mock 演示模式。"


def _get_agent(mode: str) -> tuple[PersonalAssistantAgent, str, str]:
    global AGENT, CURRENT_MODE
    if AGENT is None or CURRENT_MODE != mode:
        AGENT, CURRENT_MODE, status = _create_agent(mode)
        return AGENT, CURRENT_MODE, status
    return AGENT, CURRENT_MODE, f"当前使用 {CURRENT_MODE}。"


def _memory_markdown(agent: PersonalAssistantAgent | None = None) -> str:
    if agent is None:
        agent, _, _ = _get_agent("Mock 演示")
    entries = agent.long_term_memory.entries
    if not entries:
        return "### 长期记忆\n暂无长期记忆。"

    cards = ["### 长期记忆"]
    for entry in reversed(entries[-8:]):
        cards.append(
            "\n".join(
                [
                    f"**{entry.category}**",
                    f"> {entry.content}",
                    f"`更新于 {entry.updated_at}`",
                ]
            )
        )
    return "\n".join(cards)


def _status_markdown(text: str, warning: bool = False) -> str:
    class_name = "status-warn" if warning else "status-ok"
    return f'<span class="{class_name}">{text}</span>'


def _format_answer(answer: str, react_trace: str, show_react: bool) -> str:
    if not show_react:
        return answer
    return f"{answer}\n\n**ReAct 推理过程**\n\n```text\n{react_trace}\n```"


def submit_message(
    user_input: str,
    history: list[tuple[str, str]] | None,
    mode: str,
    show_react: bool,
) -> tuple[str, list[tuple[str, str]], str, str]:
    history = history or []
    user_input = (user_input or "").strip()
    if not user_input:
        agent, actual_mode, status = _get_agent(mode)
        return "", history, _memory_markdown(agent), _status_markdown(status, warning=actual_mode != mode)

    agent, actual_mode, status = _get_agent(mode)
    try:
        result = agent.ask(user_input)
    except Exception as exc:
        response = f"运行失败：{exc}"
        return (
            "",
            history + [(user_input, response)],
            _memory_markdown(agent),
            _status_markdown("本轮运行失败。", warning=True),
        )

    response = _format_answer(result.answer, result.react_trace, show_react)
    status_text = status if actual_mode == mode else f"{status}"
    return (
        "",
        history + [(user_input, response)],
        _memory_markdown(agent),
        _status_markdown(status_text),
    )


def switch_mode(mode: str) -> tuple[list[tuple[str, str]], str, str, object]:
    agent, actual_mode, status = _get_agent(mode)
    return [], _memory_markdown(agent), _status_markdown(status, warning=actual_mode != mode), gr.update(value=actual_mode)


def clear_short_memory(mode: str) -> tuple[list[tuple[str, str]], str]:
    agent, _, _ = _get_agent(mode)
    agent.chat_history.clear()
    return [], _status_markdown("短期对话记忆已清空，长期记忆仍保留。")


def refresh_memory(mode: str) -> str:
    agent, _, _ = _get_agent(mode)
    agent.long_term_memory.load()
    return _memory_markdown(agent)


def use_example(example: str) -> str:
    return example


def build_demo() -> gr.Blocks:
    Path("data").mkdir(exist_ok=True)
    initial_agent, initial_mode, initial_status = _get_agent("智谱 API")

    with gr.Blocks(css=CSS, title="个人智能助手 Agent", theme=gr.themes.Soft(primary_hue="blue")) as demo:
        gr.HTML(
            """
            <div id="hero">
              <h1>个人对话式智能助手 Agent</h1>
              <p>工具调用、短期上下文、长期偏好记忆和 ReAct 过程统一展示。</p>
            </div>
            """
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=7, elem_classes=["panel"]):
                chatbot = gr.Chatbot(
                    label="对话",
                    height=540,
                    show_copy_button=True,
                    bubble_full_width=False,
                )
                with gr.Row():
                    user_box = gr.Textbox(
                        label="输入消息",
                        placeholder="例如：帮我算 256 × 1024",
                        lines=3,
                        scale=8,
                    )
                    send_btn = gr.Button("发送", variant="primary", scale=1)

                with gr.Row():
                    for text in [
                        "帮我算 256 × 1024",
                        "我喜欢喝咖啡，不爱喝茶，尽量给我推荐冷饮",
                        "给我推荐一款饮品",
                        "刚才算的那个结果再乘以 2",
                        "今天是几号？帮我按照我的喜好推荐一杯今天适合喝的东西",
                    ]:
                        gr.Button(text, size="sm").click(use_example, inputs=[gr.State(text)], outputs=[user_box])

            with gr.Column(scale=3):
                with gr.Group(elem_classes=["panel"]):
                    mode_radio = gr.Radio(
                        choices=["智谱 API", "Mock 演示"],
                        value=initial_mode,
                        label="运行模式",
                    )
                    show_react = gr.Checkbox(value=True, label="显示 ReAct 推理过程")
                    status = gr.HTML(value=_status_markdown(initial_status, warning=initial_mode != "智谱 API"), label="状态")
                    clear_btn = gr.Button("清空短期对话")
                    refresh_btn = gr.Button("刷新长期记忆")

                with gr.Group(elem_classes=["panel"]):
                    gr.Markdown("### 内置工具\n- 计算器：数学表达式计算\n- 时间工具：日期、时间、倒计时\n- 信息检索：联网搜索或模拟搜索")
                    memory_md = gr.Markdown(value=_memory_markdown(initial_agent))

        send_inputs = [user_box, chatbot, mode_radio, show_react]
        send_outputs = [user_box, chatbot, memory_md, status]
        user_box.submit(submit_message, inputs=send_inputs, outputs=send_outputs, show_progress="minimal")
        send_btn.click(submit_message, inputs=send_inputs, outputs=send_outputs, show_progress="minimal")
        mode_radio.change(switch_mode, inputs=[mode_radio], outputs=[chatbot, memory_md, status, mode_radio], show_progress="minimal")
        clear_btn.click(clear_short_memory, inputs=[mode_radio], outputs=[chatbot, status], show_progress="minimal")
        refresh_btn.click(refresh_memory, inputs=[mode_radio], outputs=[memory_md], show_progress="minimal")

    return demo


def _find_available_port(start_port: int = 7860, attempts: int = 50) -> int:
    for port in range(start_port, start_port + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise OSError(f"无法在 {start_port}-{start_port + attempts - 1} 范围内找到可用端口。")


if __name__ == "__main__":
    # Gradio 4.x may reject local launch when its localhost HEAD check is blocked
    # or returns a non-200 status in managed environments. The server itself still
    # runs normally, so we bypass only that preflight check for this local app.
    import gradio.networking as networking

    networking.url_ok = lambda _url: True
    port = int(os.getenv("GRADIO_SERVER_PORT", _find_available_port()))
    print(f"Gradio 界面启动中：http://127.0.0.1:{port}")
    build_demo().launch(server_name="127.0.0.1", server_port=port, inbrowser=False)
