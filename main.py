from __future__ import annotations

import argparse

from personal_assistant.agent import PersonalAssistantAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="个人对话式智能助手 Agent")
    parser.add_argument("--mock", action="store_true", help="不调用大模型 API，使用本地规则演示工具和记忆效果。")
    parser.add_argument("--no-react", action="store_true", help="只输出回答，不显示 ReAct 过程。")
    parser.add_argument("--memory-path", default="data/long_term_memory.json", help="长期记忆 JSON 文件路径。")
    args = parser.parse_args()

    assistant = PersonalAssistantAgent(
        memory_path=args.memory_path,
        show_react=not args.no_react,
        mock=args.mock,
    )
    print("个人对话式智能助手已启动。输入 exit / quit 退出。")
    while True:
        try:
            user_input = input("\n用户：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break
        if user_input.lower() in {"exit", "quit", "q"}:
            print("已退出。")
            break
        if not user_input:
            continue

        result = assistant.ask(user_input)
        if assistant.show_react:
            print(f"Agent（ReAct 完整流程）：\n{result.react_trace}")
        else:
            print(f"Agent：{result.answer}")


if __name__ == "__main__":
    main()

