from __future__ import annotations

import os
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from dotenv import load_dotenv

from personal_assistant.agent import PersonalAssistantAgent


class AssistantGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        load_dotenv()
        self.title("个人智能助手 Agent")
        self.geometry("1160x760")
        self.minsize(980, 640)
        self.configure(bg="#EEF2F7")

        self.memory_path = "data/long_term_memory.json"
        self.result_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.agent = self._create_agent()

        self.mode_var = tk.StringVar(value="Mock" if self.agent.mock else "API")
        self.react_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="就绪")
        self.memory_var = tk.StringVar(value="")

        self._build_layout()
        self._refresh_memory_panel()
        self.after(100, self._poll_result_queue)

    def _create_agent(self, mock: bool | None = None) -> PersonalAssistantAgent:
        if mock is None:
            mock = not bool(os.getenv("ZHIPUAI_API_KEY") or os.getenv("ZHIPU_API_KEY"))
        try:
            return PersonalAssistantAgent(memory_path=self.memory_path, show_react=True, mock=mock)
        except RuntimeError as exc:
            messagebox.showwarning("API 配置缺失", f"{exc}\n\n已切换到 Mock 演示模式。")
            return PersonalAssistantAgent(memory_path=self.memory_path, show_react=True, mock=True)

    def _build_layout(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        sidebar = tk.Frame(self, bg="#111827", width=292)
        sidebar.grid(row=0, column=0, sticky="ns")
        sidebar.grid_propagate(False)
        sidebar.grid_rowconfigure(4, weight=1)

        tk.Label(
            sidebar,
            text="Personal Agent",
            bg="#111827",
            fg="#F9FAFB",
            font=("PingFang SC", 22, "bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=24, pady=(28, 4))
        tk.Label(
            sidebar,
            text="工具调用 · 上下文记忆 · 长期记忆",
            bg="#111827",
            fg="#9CA3AF",
            font=("PingFang SC", 12),
            anchor="w",
        ).grid(row=1, column=0, sticky="ew", padx=24, pady=(0, 24))

        mode_card = self._sidebar_card(sidebar)
        mode_card.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 14))
        tk.Label(mode_card, text="运行模式", bg="#1F2937", fg="#D1D5DB", font=("PingFang SC", 12, "bold")).pack(anchor="w")
        mode_row = tk.Frame(mode_card, bg="#1F2937")
        mode_row.pack(fill="x", pady=(12, 0))
        self._mode_button(mode_row, "Mock").pack(side="left", expand=True, fill="x", padx=(0, 6))
        self._mode_button(mode_row, "API").pack(side="left", expand=True, fill="x", padx=(6, 0))

        tools_card = self._sidebar_card(sidebar)
        tools_card.grid(row=3, column=0, sticky="ew", padx=18, pady=(0, 14))
        tk.Label(tools_card, text="内置工具", bg="#1F2937", fg="#D1D5DB", font=("PingFang SC", 12, "bold")).pack(anchor="w")
        for label, color in [("计算器", "#60A5FA"), ("时间 / 日期 / 倒计时", "#34D399"), ("联网搜索 / 模拟搜索", "#FBBF24")]:
            self._tool_badge(tools_card, label, color).pack(anchor="w", pady=(10, 0))

        memory_card = self._sidebar_card(sidebar)
        memory_card.grid(row=4, column=0, sticky="nsew", padx=18, pady=(0, 18))
        tk.Label(memory_card, text="长期记忆", bg="#1F2937", fg="#D1D5DB", font=("PingFang SC", 12, "bold")).pack(anchor="w")
        self.memory_text = tk.Text(
            memory_card,
            height=12,
            bg="#111827",
            fg="#E5E7EB",
            insertbackground="#E5E7EB",
            relief="flat",
            wrap="word",
            padx=12,
            pady=10,
            font=("PingFang SC", 12),
        )
        self.memory_text.pack(fill="both", expand=True, pady=(12, 12))
        self.memory_text.configure(state="disabled")
        self._secondary_button(memory_card, "刷新记忆", self._refresh_memory_panel).pack(fill="x")

        main = tk.Frame(self, bg="#EEF2F7")
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(1, weight=1)

        header = tk.Frame(main, bg="#EEF2F7")
        header.grid(row=0, column=0, sticky="ew", padx=28, pady=(24, 14))
        header.grid_columnconfigure(0, weight=1)
        tk.Label(
            header,
            text="对话式智能助手",
            bg="#EEF2F7",
            fg="#111827",
            font=("PingFang SC", 24, "bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="w")
        tk.Label(
            header,
            textvariable=self.status_var,
            bg="#EEF2F7",
            fg="#64748B",
            font=("PingFang SC", 12),
            anchor="e",
        ).grid(row=0, column=1, sticky="e")

        chat_shell = tk.Frame(main, bg="#FFFFFF", highlightbackground="#DDE5EF", highlightthickness=1)
        chat_shell.grid(row=1, column=0, sticky="nsew", padx=28, pady=(0, 18))
        chat_shell.grid_columnconfigure(0, weight=1)
        chat_shell.grid_rowconfigure(0, weight=1)

        self.chat_canvas = tk.Canvas(chat_shell, bg="#FFFFFF", highlightthickness=0)
        self.chat_scrollbar = tk.Scrollbar(chat_shell, orient="vertical", command=self.chat_canvas.yview)
        self.chat_frame = tk.Frame(self.chat_canvas, bg="#FFFFFF")
        self.chat_window = self.chat_canvas.create_window((0, 0), window=self.chat_frame, anchor="nw")
        self.chat_canvas.configure(yscrollcommand=self.chat_scrollbar.set)
        self.chat_canvas.grid(row=0, column=0, sticky="nsew")
        self.chat_scrollbar.grid(row=0, column=1, sticky="ns")
        self.chat_frame.bind("<Configure>", self._on_chat_configure)
        self.chat_canvas.bind("<Configure>", self._on_canvas_configure)

        self._add_system_message("输入问题开始对话。示例：帮我算 256 × 1024；我喜欢喝咖啡，不爱喝茶；给我推荐一款饮品。")

        composer = tk.Frame(main, bg="#EEF2F7")
        composer.grid(row=2, column=0, sticky="ew", padx=28, pady=(0, 24))
        composer.grid_columnconfigure(0, weight=1)

        input_shell = tk.Frame(composer, bg="#FFFFFF", highlightbackground="#CBD5E1", highlightthickness=1)
        input_shell.grid(row=0, column=0, sticky="ew", padx=(0, 12))
        input_shell.grid_columnconfigure(0, weight=1)
        self.input_box = tk.Text(
            input_shell,
            height=3,
            bg="#FFFFFF",
            fg="#111827",
            insertbackground="#111827",
            relief="flat",
            wrap="word",
            padx=14,
            pady=12,
            font=("PingFang SC", 13),
        )
        self.input_box.grid(row=0, column=0, sticky="ew")
        self.input_box.bind("<Command-Return>", lambda _event: self._send_message())
        self.input_box.bind("<Control-Return>", lambda _event: self._send_message())

        controls = tk.Frame(composer, bg="#EEF2F7")
        controls.grid(row=0, column=1, sticky="nsew")
        self.send_button = self._primary_button(controls, "发送", self._send_message)
        self.send_button.pack(fill="x", pady=(0, 8))
        self.react_check = tk.Checkbutton(
            controls,
            text="显示 ReAct",
            variable=self.react_var,
            bg="#EEF2F7",
            fg="#334155",
            activebackground="#EEF2F7",
            selectcolor="#FFFFFF",
            font=("PingFang SC", 12),
        )
        self.react_check.pack(anchor="w")
        self._secondary_button(controls, "清空对话", self._clear_chat).pack(fill="x", pady=(8, 0))

    def _sidebar_card(self, parent: tk.Widget) -> tk.Frame:
        return tk.Frame(parent, bg="#1F2937", padx=16, pady=16)

    def _tool_badge(self, parent: tk.Widget, text: str, color: str) -> tk.Frame:
        frame = tk.Frame(parent, bg="#1F2937")
        dot = tk.Canvas(frame, width=10, height=10, bg="#1F2937", highlightthickness=0)
        dot.create_oval(1, 1, 9, 9, fill=color, outline=color)
        dot.pack(side="left", padx=(0, 8))
        tk.Label(frame, text=text, bg="#1F2937", fg="#E5E7EB", font=("PingFang SC", 12)).pack(side="left")
        return frame

    def _mode_button(self, parent: tk.Widget, mode: str) -> tk.Button:
        is_active = self.mode_var.get() == mode
        return tk.Button(
            parent,
            text=mode,
            command=lambda: self._switch_mode(mode),
            relief="flat",
            bd=0,
            bg="#2563EB" if is_active else "#374151",
            fg="#FFFFFF" if is_active else "#D1D5DB",
            activebackground="#1D4ED8",
            activeforeground="#FFFFFF",
            font=("PingFang SC", 12, "bold"),
            padx=10,
            pady=9,
            cursor="hand2",
        )

    def _primary_button(self, parent: tk.Widget, text: str, command) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            relief="flat",
            bd=0,
            bg="#2563EB",
            fg="#FFFFFF",
            activebackground="#1D4ED8",
            activeforeground="#FFFFFF",
            font=("PingFang SC", 13, "bold"),
            padx=22,
            pady=13,
            cursor="hand2",
        )

    def _secondary_button(self, parent: tk.Widget, text: str, command) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            relief="flat",
            bd=0,
            bg="#E2E8F0",
            fg="#334155",
            activebackground="#CBD5E1",
            activeforeground="#0F172A",
            font=("PingFang SC", 12, "bold"),
            padx=14,
            pady=10,
            cursor="hand2",
        )

    def _switch_mode(self, mode: str) -> None:
        if mode == self.mode_var.get():
            return
        self.agent = self._create_agent(mock=(mode == "Mock"))
        actual_mode = "Mock" if self.agent.mock else "API"
        self.mode_var.set(actual_mode)
        self.status_var.set(f"已切换到 {actual_mode} 模式")
        self._rebuild_sidebar_mode_buttons()

    def _rebuild_sidebar_mode_buttons(self) -> None:
        # Keep the implementation simple: rebuild the window so button colors stay truthful.
        for widget in self.winfo_children():
            widget.destroy()
        self._build_layout()
        self._refresh_memory_panel()

    def _refresh_memory_panel(self) -> None:
        entries = self.agent.long_term_memory.entries
        if entries:
            text = "\n\n".join(f"[{entry.category}]\n{entry.content}" for entry in reversed(entries[-8:]))
        else:
            text = "暂无长期记忆。"
        self.memory_text.configure(state="normal")
        self.memory_text.delete("1.0", "end")
        self.memory_text.insert("1.0", text)
        self.memory_text.configure(state="disabled")

    def _send_message(self) -> None:
        user_input = self.input_box.get("1.0", "end").strip()
        if not user_input:
            return
        self.input_box.delete("1.0", "end")
        self._add_chat_bubble("用户", user_input, align="right")
        self._set_busy(True)
        thread = threading.Thread(target=self._run_agent, args=(user_input,), daemon=True)
        thread.start()

    def _run_agent(self, user_input: str) -> None:
        try:
            result = self.agent.ask(user_input)
            self.result_queue.put(("ok", result))
        except Exception as exc:
            self.result_queue.put(("error", exc))

    def _poll_result_queue(self) -> None:
        try:
            status, payload = self.result_queue.get_nowait()
        except queue.Empty:
            self.after(100, self._poll_result_queue)
            return

        self._set_busy(False)
        if status == "ok":
            result = payload
            self._add_chat_bubble("Agent", result.answer, align="left")
            if self.react_var.get():
                self._add_trace_card(result.react_trace)
            self._refresh_memory_panel()
            self.status_var.set("已完成")
        else:
            self.status_var.set("出错")
            messagebox.showerror("运行失败", str(payload))
        self.after(100, self._poll_result_queue)

    def _set_busy(self, busy: bool) -> None:
        self.send_button.configure(state="disabled" if busy else "normal")
        self.status_var.set("Agent 正在思考..." if busy else "就绪")

    def _clear_chat(self) -> None:
        self.agent.chat_history.clear()
        for child in self.chat_frame.winfo_children():
            child.destroy()
        self._add_system_message("短期对话已清空，长期记忆仍保留。")
        self.status_var.set("已清空短期对话")

    def _add_system_message(self, text: str) -> None:
        label = tk.Label(
            self.chat_frame,
            text=text,
            bg="#F1F5F9",
            fg="#64748B",
            wraplength=680,
            justify="left",
            font=("PingFang SC", 12),
            padx=14,
            pady=10,
        )
        label.pack(anchor="center", pady=(18, 8), padx=22)
        self._scroll_to_bottom()

    def _add_chat_bubble(self, sender: str, text: str, align: str) -> None:
        outer = tk.Frame(self.chat_frame, bg="#FFFFFF")
        outer.pack(fill="x", padx=22, pady=10)
        bubble_bg = "#2563EB" if align == "right" else "#F8FAFC"
        text_fg = "#FFFFFF" if align == "right" else "#111827"
        border = "#2563EB" if align == "right" else "#E2E8F0"
        inner = tk.Frame(outer, bg=bubble_bg, padx=16, pady=12, highlightbackground=border, highlightthickness=1)
        inner.pack(anchor="e" if align == "right" else "w")
        tk.Label(
            inner,
            text=sender,
            bg=bubble_bg,
            fg="#DBEAFE" if align == "right" else "#64748B",
            font=("PingFang SC", 10, "bold"),
            anchor="w",
        ).pack(anchor="w")
        tk.Label(
            inner,
            text=text,
            bg=bubble_bg,
            fg=text_fg,
            wraplength=620,
            justify="left",
            font=("PingFang SC", 13),
        ).pack(anchor="w", pady=(5, 0))
        self._scroll_to_bottom()

    def _add_trace_card(self, text: str) -> None:
        outer = tk.Frame(self.chat_frame, bg="#FFFFFF")
        outer.pack(fill="x", padx=22, pady=(0, 12))
        card = tk.Frame(outer, bg="#FFFBEB", padx=14, pady=12, highlightbackground="#FDE68A", highlightthickness=1)
        card.pack(anchor="w", fill="x")
        tk.Label(
            card,
            text="ReAct 过程",
            bg="#FFFBEB",
            fg="#92400E",
            font=("PingFang SC", 11, "bold"),
        ).pack(anchor="w")
        tk.Label(
            card,
            text=text,
            bg="#FFFBEB",
            fg="#78350F",
            wraplength=720,
            justify="left",
            font=("PingFang SC", 12),
        ).pack(anchor="w", pady=(6, 0))
        self._scroll_to_bottom()

    def _on_chat_configure(self, _event) -> None:
        self.chat_canvas.configure(scrollregion=self.chat_canvas.bbox("all"))

    def _on_canvas_configure(self, event) -> None:
        self.chat_canvas.itemconfigure(self.chat_window, width=event.width)

    def _scroll_to_bottom(self) -> None:
        self.after_idle(lambda: self.chat_canvas.yview_moveto(1.0))


def main() -> None:
    Path("data").mkdir(exist_ok=True)
    app = AssistantGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
