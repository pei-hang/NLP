# 个人对话式智能助手 Agent

技术栈：Python + LangChain + 智谱 OpenAI-compatible API。

## 功能

- 工具调用：计算器、日期时间/倒计时、联网搜索。
- 短期记忆：进程内保存多轮聊天上下文，支持“刚才那个结果再乘以 2”。
- 长期记忆：自动提取用户偏好、习惯和关键信息，持久化到 `data/long_term_memory.json`。
- ReAct 展示：CLI 默认输出“思考 - 行动 - 观察 - 回答”的可解释流程。

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

然后在 `.env` 中填写：

```bash
ZHIPUAI_API_KEY=你的智谱API密钥
```

## 运行

调用真实智谱 API：

```bash
python main.py
```

无 API Key 的本地演示模式：

```bash
python main.py --mock
```

只看最终回答，不显示 ReAct 流程：

```bash
python main.py --no-react
```

启动图形界面：

```bash
python app_gui.py
```

GUI 会在没有 API Key 时自动进入 Mock 演示模式；填写 `.env` 后可切换到 API 模式。

启动 Gradio Web 可视化界面：

```bash
python app_gradio.py
```

默认地址是 `http://127.0.0.1:7860`。没有 API Key 时会自动进入 Mock 演示模式。

## 示例

```text
用户：帮我算 256 × 1024
Agent：调用 calculator，返回 262144

用户：我喜欢喝冰红茶，不爱牛奶，尽量给我推荐冷饮
Agent：写入长期记忆

用户：给我推荐一款饮品
Agent：读取长期记忆，推荐冷萃冰咖啡

用户：刚才算的那个结果再乘以 2
Agent：读取短期对话记忆，继续计算 262144 * 2
```

## 目录

```text
personal_assistant/
  agent.py      # LangChain Agent 编排、短期记忆、ReAct 展示
  memory.py     # 长期记忆持久化与偏好抽取
  tools.py      # calculator / time_tool / web_search
main.py         # 命令行入口
app_gui.py      # 桌面图形界面入口
app_gradio.py   # Gradio Web 可视化界面入口
```
