from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
from uuid import uuid4


@dataclass
class MemoryEntry:
    id: str
    category: str
    content: str
    keywords: list[str]
    created_at: str
    updated_at: str


class LongTermMemoryStore:
    """Small persistent memory store for user preferences and key facts."""

    def __init__(self, path: str | Path = "data/long_term_memory.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: list[MemoryEntry] = []
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self.entries = []
            return
        raw_text = self.path.read_text(encoding="utf-8", errors="replace").strip()
        if not raw_text:
            self.entries = []
            return
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            backup_path = self.path.with_suffix(f"{self.path.suffix}.bak")
            self.path.replace(backup_path)
            self.entries = []
            return
        self.entries = [MemoryEntry(**item) for item in payload]
        if self._deduplicate_preference_conflicts():
            self.save()

    def save(self) -> None:
        payload = [_sanitize_payload(asdict(entry)) for entry in self.entries]
        tmp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(self.path)

    def add_or_update(self, category: str, content: str, keywords: Iterable[str]) -> None:
        clean_content = _strip_surrogates(content).strip()
        clean_keywords = sorted(
            {
                _strip_surrogates(word).strip().lower()
                for word in keywords
                if _strip_surrogates(word).strip()
            }
        )
        if not clean_content:
            return
        now = datetime.now().isoformat(timespec="seconds")
        self._remove_opposite_preference(category, clean_content)
        for entry in self.entries:
            if entry.category == category and entry.content == clean_content:
                entry.keywords = sorted(set(entry.keywords) | set(clean_keywords))
                entry.updated_at = now
                self.save()
                return
        self.entries.append(
            MemoryEntry(
                id=uuid4().hex,
                category=category,
                content=clean_content,
                keywords=clean_keywords,
                created_at=now,
                updated_at=now,
            )
        )
        self.save()

    def _remove_opposite_preference(self, category: str, content: str) -> None:
        if category not in {"preference_like", "preference_dislike"}:
            return
        key = _preference_key(category, content)
        if not key:
            return
        opposite_category = "preference_dislike" if category == "preference_like" else "preference_like"
        self.entries = [
            entry
            for entry in self.entries
            if not (entry.category == opposite_category and _preference_key(entry.category, entry.content) == key)
        ]

    def _deduplicate_preference_conflicts(self) -> bool:
        original_count = len(self.entries)
        resolved: list[MemoryEntry] = []
        for entry in sorted(self.entries, key=lambda item: item.updated_at):
            key = _preference_key(entry.category, entry.content)
            if key and entry.category in {"preference_like", "preference_dislike"}:
                opposite = "preference_dislike" if entry.category == "preference_like" else "preference_like"
                resolved = [
                    item
                    for item in resolved
                    if not (item.category == opposite and _preference_key(item.category, item.content) == key)
                ]
            resolved.append(entry)
        if len(resolved) != original_count:
            self.entries = resolved
            return True
        self.entries = sorted(resolved, key=lambda item: item.created_at)
        return False

    def retrieve(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        query_terms = set(_tokenize(query))
        scored: list[tuple[int, MemoryEntry]] = []
        for entry in self.entries:
            haystack = set(entry.keywords) | set(_tokenize(entry.content))
            score = len(query_terms & haystack)
            if score:
                scored.append((score, entry))
        scored.sort(key=lambda item: (item[0], item[1].updated_at), reverse=True)
        selected: list[MemoryEntry] = [entry for _, entry in scored[:limit]]
        selected_ids = {entry.id for entry in selected}
        for entry in reversed(self.entries):
            if len(selected) >= limit:
                break
            if entry.id not in selected_ids:
                selected.append(entry)
                selected_ids.add(entry.id)
        return selected

    def format_for_prompt(self, query: str) -> str:
        memories = self.retrieve(query)
        if not memories:
            return "暂无长期记忆。"
        return "\n".join(f"- [{item.category}] {item.content}" for item in memories)

    def update_from_user_message(self, message: str) -> list[str]:
        extracted: list[str] = []
        message = _strip_surrogates(message)
        clauses = [clause.strip() for clause in re.split(r"[，,。！？\n；;]", message) if clause.strip()]
        for clause in clauses:
            if _is_task_request_clause(clause):
                continue
            for pattern, category, label in [
                (r"^(?:我|本人)?(?:平时)?(?<!不)(?:喜欢|爱|偏好)(.+)", "preference_like", "喜欢"),
                (r"^(?:我|本人)?(?:不喜欢|不爱|不喝|讨厌)(.+)", "preference_dislike", "不喜欢"),
                (r"^(?:尽量|希望|以后|优先)(.+)", "preference_rule", "偏好规则"),
                (r"^(?:记住|请记住)(.+)", "user_fact", "用户信息"),
            ]:
                match = re.search(pattern, clause)
                if not match:
                    continue
                raw = _clean_memory_fragment(match.group(1))
                if raw:
                    content = f"用户{label}{raw}"
                    self.add_or_update(category, content, _tokenize(raw))
                    extracted.append(content)
        return extracted


def _clean_memory_fragment(text: str) -> str:
    text = _strip_surrogates(text)
    text = re.sub(r"^[：:，,\s]+", "", text.strip())
    text = re.sub(r"^(?:你|帮我|给我)", "", text).strip()
    return text


def _is_task_request_clause(clause: str) -> bool:
    task_prefixes = ("给我", "帮我", "请", "麻烦", "帮", "为我", "替我")
    task_verbs = (
        "写",
        "生成",
        "推荐",
        "总结",
        "翻译",
        "查询",
        "搜索",
        "计算",
        "做",
        "创建",
        "制定",
        "改",
        "润色",
    )
    return clause.startswith(task_prefixes) and any(verb in clause for verb in task_verbs)


def _tokenize(text: str) -> list[str]:
    text = _strip_surrogates(text)
    chinese_terms = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    latin_terms = re.findall(r"[A-Za-z0-9_]+", text.lower())
    beverage_terms = [
        word
        for word in ("咖啡", "茶", "冷饮", "热饮", "饮品", "推荐", "夏天", "冰", "美式", "拿铁")
        if word in text
    ]
    return chinese_terms + latin_terms + beverage_terms


def _preference_key(category: str, content: str) -> str:
    if category == "preference_like":
        return re.sub(r"^用户喜欢", "", content).strip()
    if category == "preference_dislike":
        return re.sub(r"^用户不喜欢", "", content).strip()
    return ""


def _strip_surrogates(text: str) -> str:
    return text.encode("utf-8", errors="ignore").decode("utf-8")


def _sanitize_payload(value):
    if isinstance(value, str):
        return _strip_surrogates(value)
    if isinstance(value, list):
        return [_sanitize_payload(item) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize_payload(item) for key, item in value.items()}
    return value
