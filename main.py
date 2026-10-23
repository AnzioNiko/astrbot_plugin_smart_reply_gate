from __future__ import annotations

import asyncio
import inspect
import json
import random
import re
import time
from pathlib import Path
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register

from .reply_judge import ReplyJudge


PLUGIN_NAME = "astrbot_plugin_smart_reply_gate"
META_ATTR = "_smart_reply_gate_meta"


@register(
    PLUGIN_NAME,
    "Anzio",
    "通过关键词、正则和群聊主动抽样触发带判断门控的回复",
    "v0.1.0",
)
class SmartReplyGatePlugin(Star):
    def __init__(self, context: Context, config: dict[str, Any]):
        super().__init__(context)
        self.config = config or {}
        self._remove_legacy_typesafe_api_key()
        self._migrate_rule_list_config()
        self.judge = ReplyJudge(context, self.config)
        self.keywords = self._parse_lines(self.config.get("keywords", ""))
        self.regex_patterns: list[re.Pattern[str]] = []
        self._compile_patterns(self._parse_lines(self.config.get("regex_rules", "")))
        self.fixed_cooldowns: dict[str, float] = {}
        self.followup_windows: dict[tuple[str, str], float] = {}
        self.energy: dict[str, dict[str, float]] = {}
        self.energy_lock = asyncio.Lock()
        self.data_dir = self._get_data_dir()
        self.energy_file = self.data_dir / "group_energy.json"
        self._load_energy()

    def _remove_legacy_typesafe_api_key(self):
        """移除旧版插件设置中的密钥字段，并尽可能持久化清理结果。"""
        if "typesafe_api_key" not in self.config:
            return

        self.config.pop("typesafe_api_key", None)
        save_config = getattr(self.config, "save_config", None)
        if callable(save_config):
            try:
                save_config()
            except Exception:
                logger.warning(
                    "[智能回复门控] 已停止读取旧版 TypeSafe 密钥设置，但未能自动清理持久化配置；"
                    "请在插件配置中删除旧的 typesafe_api_key 字段。"
                )
            else:
                logger.info(
                    "[智能回复门控] 已清理旧版 TypeSafe 密钥设置；现在只从环境变量 TYPESAFE_API_KEY 读取密钥。"
                )
        else:
            logger.warning(
                "[智能回复门控] 已停止读取旧版 TypeSafe 密钥设置；"
                "请在插件配置中删除旧的 typesafe_api_key 字段。"
            )

    def _migrate_rule_list_config(self):
        """将旧版换行文本规则迁移为 AstrBot 列表配置。"""
        migrated_keys = []
        for key in ("keywords", "regex_rules"):
            value = self.config.get(key, [])
            if isinstance(value, str):
                self.config[key] = self._parse_lines(value)
                migrated_keys.append(key)

        if not migrated_keys:
            return

        save_config = getattr(self.config, "save_config", None)
        if not callable(save_config):
            logger.warning(
                "[智能回复门控] 已将旧版关键词/正则配置转换为列表，但当前配置对象不支持持久化；"
                "请在插件设置中保存一次配置。"
            )
            return

        try:
            save_config()
        except Exception as exc:
            logger.error(
                "[智能回复门控] 旧版关键词/正则配置已转换为列表，但保存失败：%s",
                exc,
            )
        else:
            logger.info(
                "[智能回复门控] 已将旧版关键词/正则配置迁移为列表：%s",
                ", ".join(migrated_keys),
            )

    @filter.event_message_type(filter.EventMessageType.ALL, priority=-1000)
    async def on_message(self, event: AstrMessageEvent):
        """先排除非回复事件，再按触发规则决定是否请求判断器。"""
        sender_id = self._sender_id(event)
        self_id = str(event.get_self_id() or "").strip()
        if sender_id and self_id and sender_id == self_id:
            logger.info(
                "[智能回复门控] 会话=%s 来源=机器人自身消息 跳过插件处理",
                self._session(event),
            )
            return

        get_extra = getattr(event, "get_extra", None)
        if callable(get_extra) and get_extra("handlers_parsed_params", {}):
            # 命令 handler 已经由 AstrBot 匹配并执行；阻止后续默认 LLM 再把命令文本当聊天回复。
            # should_call_llm 只控制默认 LLM，不会停止其他插件的命令 handler。
            event.should_call_llm(True)
            logger.info(
                "[智能回复门控] 会话=%s 来源=其他插件指令 跳过判断并阻止默认主模型，保留命令处理",
                self._session(event),
            )
            return

        is_private = bool(event.is_private_chat())
        is_at_or_wake_command = bool(getattr(event, "is_at_or_wake_command", False))
        if is_private and is_at_or_wake_command:
            logger.info(
                "[智能回复门控] 会话=%s 来源=AstrBot唤醒或外部插件触发 跳过插件处理",
                self._session(event),
            )
            return

        followup_enabled = self._bool_config("followup_enabled", True)
        if not followup_enabled:
            self.followup_windows.clear()

        if is_private:
            text = str(getattr(event, "message_str", "") or "")
            explicit_source = self._match_explicit(text)
            if not self._bool_config("private_rules_enabled", False):
                return
            if not explicit_source:
                event.should_call_llm(True)
                logger.info("[智能回复门控] 会话=%s 来源=私聊规则 结果=未命中，阻止默认回复", self._session(event))
                return
            await self._judge_and_route(event, explicit_source, explicit=True)
            return

        group_id = self._group_id(event)
        if not group_id:
            return

        sender_id = self._sender_id(event)
        now = time.time()
        is_at_bot = self._has_at_bot(event)
        has_at = self._has_any_at(event)
        has_quote = self._has_quote(event)
        is_quote_to_bot = self._has_quote_to_bot(event)
        if is_at_or_wake_command and not is_at_bot and not has_at and not has_quote:
            logger.info(
                "[智能回复门控] 会话=%s 来源=AstrBot唤醒或外部插件触发 跳过插件处理",
                self._session(event),
            )
            return

        text = str(getattr(event, "message_str", "") or "")
        explicit_source = self._match_explicit(text)
        if followup_enabled and sender_id:
            followup_expiry = self.followup_windows.get((group_id, sender_id), 0.0)
            if followup_expiry > now:
                await self._route_followup(event, group_id, sender_id, followup_expiry)
                return
            if followup_expiry:
                self.followup_windows.pop((group_id, sender_id), None)

        # 关键词/正则规则始终优先于 @BOT 和对机器人消息的引用。
        if explicit_source:
            await self._judge_and_route(event, explicit_source, explicit=True)
            return

        if is_at_bot or is_quote_to_bot:
            if self._bool_config("judge_mentions_and_quotes", True):
                if is_at_bot and is_quote_to_bot:
                    source = "机器人@与引用机器人消息"
                elif is_at_bot:
                    source = "机器人@消息"
                else:
                    source = "引用机器人消息"
                await self._judge_and_route(event, source, explicit=True)
            else:
                self._set_meta(event, {"ordinary_mention": True, "group_id": group_id})
            return

        # @其他人或引用他人消息与普通群消息一样，先经过主动回复抽样和限频。
        if (
            not sender_id
            or not self._bool_config("active_reply_enabled", False)
            or not self._group_allowed(group_id)
        ):
            return

        probability = max(
            0.0, min(100.0, self._float_config("active_probability", 10.0))
        )
        if random.random() * 100.0 >= probability:
            event.should_call_llm(True)
            logger.info(
                "[智能回复门控] 会话=%s 来源=主动抽样 结果=未抽中 概率=%.1f%% 跳过判断和主模型",
                self._session(event),
                probability,
            )
            return
        logger.info(
            "[智能回复门控] 会话=%s 来源=主动抽样 结果=抽中 概率=%.1f%%",
            self._session(event),
            probability,
        )

        # 先抽样，再检查冷却/能量，避免为未抽中的群消息请求判断模型。
        if not await self._active_limiter_allowed(group_id, now):
            event.should_call_llm(True)
            logger.info(
                "[智能回复门控] 会话=%s 来源=主动抽样 限频未通过，跳过判断和主模型",
                self._session(event),
            )
            return

        await self._judge_and_route(event, "主动抽样", explicit=False)

    async def _judge_and_route(
        self,
        event: AstrMessageEvent,
        source: str,
        explicit: bool,
        is_followup: bool = False,
        followup_expiry: float = 0.0,
    ):
        if not self._bool_config("judge_enabled", True):
            logger.info(
                "[智能回复门控] 会话=%s 来源=%s 判断器总开关关闭 结果=直接放行主模型",
                self._session(event),
                source,
            )
            self._allow_default_reply(
                event,
                source,
                is_followup=is_followup,
                followup_expiry=followup_expiry,
            )
            return

        mode = self.judge.mode
        try:
            history, persona_prompt = await self._recent_context(event)
            message = str(getattr(event, "message_str", "") or "")
            if (
                history
                and history[-1].get("role") == "user"
                and history[-1].get("content", "").strip() == message.strip()
            ):
                history = history[:-1]
            decision = await self.judge.judge(
                message,
                history,
                umo=event.unified_msg_origin,
                persona_prompt=persona_prompt,
                log_context=f"会话={self._session(event)} 来源={source}",
            )
        except Exception as exc:
            if explicit:
                logger.error(
                    "[智能回复门控] 会话=%s 来源=%s 模式=%s 判断失败：%s；回退=放行主模型",
                    self._session(event),
                    source,
                    mode,
                    exc,
                )
                self._allow_default_reply(
                    event,
                    source,
                    is_followup=is_followup,
                    followup_expiry=followup_expiry,
                )
            else:
                logger.error(
                    "[智能回复门控] 会话=%s 来源=%s 模式=%s 判断失败：%s；回退=跳过回复",
                    self._session(event),
                    source,
                    mode,
                    exc,
                )
                event.should_call_llm(True)
            return

        score = decision.get("score")
        threshold = decision.get("threshold")
        score_text = ""
        if score is not None:
            score_text = f" 分数={score} 阈值={threshold}"
        result = "允许回复" if decision.get("allow") else "拒绝回复"
        logger.info(
            "[智能回复门控] 会话=%s 来源=%s 模式=%s 结果=%s%s",
            self._session(event),
            source,
            mode,
            result,
            score_text,
        )
        if decision.get("allow"):
            self._allow_default_reply(
                event,
                source,
                is_followup=is_followup,
                followup_expiry=followup_expiry,
            )
        else:
            event.should_call_llm(True)

    def _allow_default_reply(
        self,
        event: AstrMessageEvent,
        source: str,
        is_followup: bool = False,
        followup_expiry: float = 0.0,
    ):
        group_id = self._group_id(event)
        sender_id = self._sender_id(event)
        self._set_meta(
            event,
            {
                "plugin_trigger": True,
                "source": source,
                "group_id": group_id,
                "sender_id": sender_id,
                "sent": False,
                "followup_expiry": followup_expiry
                or self.followup_windows.get((group_id, sender_id), 0.0),
                "is_followup": is_followup,
            },
        )
        event.should_call_llm(False)
        event.is_at_or_wake_command = True

    async def _route_followup(
        self,
        event: AstrMessageEvent,
        group_id: str,
        sender_id: str,
        expiry: float,
    ):
        refreshed = False
        if self._bool_config("followup_refresh", False):
            expiry = time.time() + max(1, self._int_config("followup_seconds", 30))
            self.followup_windows[(group_id, sender_id)] = expiry
            refreshed = True

        if self._bool_config("judge_enabled", True) and self._bool_config(
            "followup_judge_enabled", True
        ):
            logger.info(
                "[智能回复门控] 会话=%s 来源=跟进窗口 结果=进入回复判断 刷新计时=%s 到期时间戳=%.0f",
                self._session(event),
                "是" if refreshed else "否",
                expiry,
            )
            await self._judge_and_route(
                event,
                "跟进窗口",
                explicit=True,
                is_followup=True,
                followup_expiry=expiry,
            )
            return

        self._set_meta(
            event,
            {
                "plugin_trigger": True,
                "source": "跟进窗口",
                "group_id": group_id,
                "sender_id": sender_id,
                "sent": False,
                "followup_expiry": expiry,
                "is_followup": True,
            },
        )
        event.should_call_llm(False)
        event.is_at_or_wake_command = True
        logger.info(
            "[智能回复门控] 会话=%s 来源=跟进窗口 结果=直接放行主模型 刷新计时=%s 到期时间戳=%.0f",
            self._session(event),
            "是" if refreshed else "否",
            expiry,
        )

    @filter.on_llm_response(priority=1000)
    async def on_llm_response(self, event: AstrMessageEvent, response: Any):
        """记录非工具调用的模型响应，供发送成功后的跟进窗口和限频状态使用。"""
        meta = self._get_meta(event)
        tool_calls = (
            getattr(response, "tools_call_name", None)
            or getattr(response, "tool_calls", None)
            or getattr(response, "function_call", None)
        )
        if tool_calls:
            return
        if meta.get("plugin_trigger"):
            meta["main_llm_response"] = True
        elif meta.get("ordinary_mention"):
            meta["ordinary_llm_response"] = True

    @filter.after_message_sent()
    async def after_message_sent(self, event: AstrMessageEvent):
        """仅在消息实际发送后扣除主动回复资源并开启跟进窗口。"""
        meta = self._get_meta(event)
        if meta.get("sent"):
            return

        plugin_trigger = bool(meta.get("plugin_trigger") and meta.get("main_llm_response"))
        ordinary_mention = bool(meta.get("ordinary_mention") and meta.get("ordinary_llm_response"))
        if not plugin_trigger and not ordinary_mention:
            return

        meta["sent"] = True
        group_id = str(meta.get("group_id") or self._group_id(event))
        sender_id = str(meta.get("sender_id") or self._sender_id(event))
        if plugin_trigger and meta.get("source") == "主动抽样":
            await self._record_active_success(group_id)

        if (
            self._bool_config("followup_enabled", True)
            and group_id
            and sender_id
        ):
            now = time.time()
            key = (group_id, sender_id)
            is_followup = bool(meta.get("is_followup"))
            if not is_followup:
                duration = max(1, self._int_config("followup_seconds", 30))
                self.followup_windows[key] = now + duration
                logger.info(
                    "[智能回复门控] 群=%s 发送者=%s 跟进窗口已开启/刷新 时长=%s秒 来源=%s",
                    group_id,
                    sender_id,
                    duration,
                    meta.get("source", "普通@BOT"),
                )

    async def _active_limiter_allowed(self, group_id: str, now: float) -> bool:
        limiter = str(self.config.get("limiter_mode", "固定冷却"))
        if limiter == "能量模式":
            async with self.energy_lock:
                state = self._recover_energy(group_id, now)
                configured_threshold = self.config.get("energy_threshold")
                default_threshold = self._float_config("energy_cost", 30.0)
                if configured_threshold is None or not str(configured_threshold).strip():
                    threshold = default_threshold
                else:
                    threshold = self._float_config("energy_threshold", default_threshold)
                threshold = max(0.0, threshold)
                if state["energy"] < threshold:
                    logger.info(
                        "[智能回复门控] 群=%s 来源=主动抽样 结果=能量不足 当前=%.1f 门槛=%.1f",
                        group_id,
                        state["energy"],
                        threshold,
                    )
                    return False
            return True

        cooldown = max(0.0, float(self._int_config("fixed_cooldown_seconds", 300)))
        last_success = self.fixed_cooldowns.get(group_id, 0.0)
        remaining = cooldown - (now - last_success)
        if remaining > 0:
            logger.info(
                "[智能回复门控] 群=%s 来源=主动抽样 结果=固定冷却中 剩余=%.0f秒",
                group_id,
                remaining,
            )
            return False
        return True

    async def _record_active_success(self, group_id: str):
        if not group_id:
            return
        if str(self.config.get("limiter_mode", "固定冷却")) == "能量模式":
            async with self.energy_lock:
                now = time.time()
                state = self._recover_energy(group_id, now)
                cost = max(0.0, self._float_config("energy_cost", 30.0))
                state["energy"] = max(0.0, state["energy"] - cost)
                state["updated_at"] = now
                self._save_energy()
                logger.info(
                    "[智能回复门控] 群=%s 能量回复成功后扣除=%.1f 当前=%.1f",
                    group_id,
                    cost,
                    state["energy"],
                )
        else:
            self.fixed_cooldowns[group_id] = time.time()
            seconds = max(0, self._int_config("fixed_cooldown_seconds", 300))
            logger.info("[智能回复门控] 群=%s 固定冷却开始 时长=%s秒", group_id, seconds)

    def _recover_energy(self, group_id: str, now: float) -> dict[str, float]:
        cap = max(1.0, min(1000.0, self._float_config("energy_capacity", 100.0)))
        period = max(1.0, min(86400.0, self._float_config("energy_recover_seconds", 30.0)))
        state = self.energy.get(group_id)
        if state is None:
            state = {"energy": cap, "updated_at": now}
            self.energy[group_id] = state
            return state
        elapsed = max(0.0, now - state.get("updated_at", now))
        state["energy"] = min(cap, max(0.0, state.get("energy", cap)) + elapsed / period)
        state["updated_at"] = now
        return state

    def _load_energy(self):
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            if not self.energy_file.exists():
                return
            payload = json.loads(self.energy_file.read_text(encoding="utf-8"))
            entries = payload.get("groups", {}) if isinstance(payload, dict) else {}
            now = time.time()
            for group_id, value in entries.items():
                if not isinstance(value, dict):
                    continue
                self.energy[str(group_id)] = {
                    "energy": max(0.0, float(value.get("energy", 100.0))),
                    "updated_at": min(now, float(value.get("updated_at", now))),
                }
                self._recover_energy(str(group_id), now)
            self._save_energy()
        except Exception as exc:
            logger.warning("[智能回复门控] 读取群聊能量文件失败：%s", exc)

    def _save_energy(self):
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            payload = {"groups": self.energy}
            temp_file = self.energy_file.with_suffix(".json.tmp")
            temp_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            temp_file.replace(self.energy_file)
        except Exception as exc:
            logger.error("[智能回复门控] 保存群聊能量文件失败：%s", exc)

    async def _recent_context(
        self, event: AstrMessageEvent
    ) -> tuple[list[dict[str, str]], str]:
        manager = getattr(self.context, "conversation_manager", None)
        conversation = None
        messages: list[dict[str, str]] = []
        umo = str(getattr(event, "unified_msg_origin", "") or "")
        try:
            if manager is not None and umo:
                cid = await manager.get_curr_conversation_id(umo)
                if cid:
                    conversation = await manager.get_conversation(umo, cid)
                history = getattr(conversation, "history", "[]")
                if isinstance(history, str):
                    history = json.loads(history)
                if isinstance(history, dict):
                    history = history.get("messages", history.get("history", []))
                if isinstance(history, list):
                    for item in history:
                        if not isinstance(item, dict):
                            continue
                        role = str(item.get("role", "")).lower()
                        if role not in {"user", "assistant"}:
                            continue
                        content = self._content_to_text(item.get("content", ""))
                        if content:
                            messages.append({"role": role, "content": content[:2000]})
        except Exception as exc:
            logger.debug("[智能回复门控] 读取最近对话失败，将只判断当前消息：%s", exc)
        persona_prompt = await self._current_persona_prompt(event, conversation, umo)
        return messages[-5:], persona_prompt

    async def _current_persona_prompt(
        self,
        event: AstrMessageEvent,
        conversation: Any,
        umo: str,
    ) -> str:
        """解析当前消息实际使用的人格提示词；失败时不阻断判断流程。"""
        persona_manager = getattr(self.context, "persona_manager", None)
        if persona_manager is None:
            return ""

        persona = None
        resolver = getattr(persona_manager, "resolve_selected_persona", None)
        resolved = False
        if callable(resolver):
            try:
                get_config = getattr(self.context, "get_config", None)
                runtime_config = get_config(umo=umo) if callable(get_config) and umo else {}
                if inspect.isawaitable(runtime_config):
                    runtime_config = await runtime_config
                provider_settings = (
                    runtime_config.get("provider_settings", {})
                    if isinstance(runtime_config, dict)
                    else {}
                )
                resolved_persona = resolver(
                    umo=umo,
                    conversation_persona_id=getattr(conversation, "persona_id", None),
                    platform_name=(
                        event.get_platform_name()
                        if callable(getattr(event, "get_platform_name", None))
                        else ""
                    ),
                    provider_settings=provider_settings,
                )
                if inspect.isawaitable(resolved_persona):
                    resolved_persona = await resolved_persona
                if isinstance(resolved_persona, (tuple, list)) and len(resolved_persona) > 1:
                    persona = resolved_persona[1]
                resolved = True
            except Exception as exc:
                logger.debug("[智能回复门控] 解析当前会话人格失败：%s", exc)

        if not resolved:
            legacy_getter = getattr(persona_manager, "get_default_persona_v3", None)
            if callable(legacy_getter):
                try:
                    persona = legacy_getter(umo)
                    if inspect.isawaitable(persona):
                        persona = await persona
                except Exception as exc:
                    logger.debug("[智能回复门控] 读取默认人格失败：%s", exc)

        if isinstance(persona, dict):
            prompt = persona.get("prompt", "")
        else:
            prompt = getattr(persona, "prompt", "")
        return str(prompt or "").strip()

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
            return " ".join(parts)
        return str(content) if content is not None else ""

    def _match_explicit(self, text: str) -> str:
        matched_keywords = [word for word in self.keywords if word in text]
        matched_regex = any(pattern.search(text) for pattern in self.regex_patterns)
        labels = []
        if matched_keywords:
            labels.append("关键词")
        if matched_regex:
            labels.append("正则")
        return "+".join(labels)

    def _compile_patterns(self, rules: list[str]):
        for rule in rules:
            try:
                self.regex_patterns.append(re.compile(rule))
            except re.error as exc:
                logger.error("[智能回复门控] 忽略无效正则 %r：%s", rule, exc)

    @staticmethod
    def _parse_lines(value: Any) -> list[str]:
        if isinstance(value, str):
            return [line.strip() for line in value.splitlines() if line.strip()]
        if isinstance(value, (list, tuple)):
            return [str(line).strip() for line in value if str(line).strip()]
        return []

    def _group_allowed(self, group_id: str) -> bool:
        configured = self.config.get("group_whitelist", [])
        ids = {str(value).strip() for value in configured if str(value).strip()} if isinstance(configured, list) else set()
        return not ids or group_id in ids

    @staticmethod
    def _has_at_bot(event: AstrMessageEvent) -> bool:
        self_id = str(event.get_self_id() or "")
        try:
            components = event.get_messages()
        except Exception:
            return False
        for component in components or []:
            kind = str(getattr(component, "type", "")).lower()
            if not kind:
                kind = component.__class__.__name__.lower()
            if "at" not in kind:
                continue
            target = getattr(component, "qq", None)
            if target is None:
                target = getattr(component, "user_id", getattr(component, "target", None))
            if target is not None and str(target) == self_id:
                return True
        return False

    @staticmethod
    def _has_any_at(event: AstrMessageEvent) -> bool:
        """检测消息是否含任意 @，用于区分 @他人与外部唤醒事件。"""
        try:
            components = event.get_messages()
        except Exception:
            return False
        for component in components or []:
            kind = str(getattr(component, "type", "") or "").strip().lower()
            if not kind:
                kind = component.__class__.__name__.strip().lower()
            if "at" in kind.rsplit(".", 1)[-1]:
                return True
        return False

    @staticmethod
    def _has_quote(event: AstrMessageEvent) -> bool:
        """检测群聊消息中是否包含引用/回复消息段。"""
        try:
            components = event.get_messages()
        except Exception:
            return False
        for component in components or []:
            kind = str(getattr(component, "type", "") or "").strip().lower()
            kind = kind.rsplit(".", 1)[-1]
            class_name = component.__class__.__name__.strip().lower()
            if kind in {"reply", "quote"} or class_name in {"reply", "quote"}:
                return True
        return False

    @staticmethod
    def _has_quote_to_bot(event: AstrMessageEvent) -> bool:
        """仅当引用段的原发送者是机器人本人时返回 True。"""
        self_id = str(event.get_self_id() or "").strip()
        if not self_id:
            return False
        try:
            components = event.get_messages()
        except Exception:
            return False
        for component in components or []:
            if isinstance(component, dict):
                kind = str(component.get("type", "") or "").strip().lower()
                sender_id = component.get("sender_id")
                if sender_id in (None, "", 0, "0"):
                    sender_id = component.get("qq")
            else:
                kind = str(getattr(component, "type", "") or "").strip().lower()
                sender_id = getattr(component, "sender_id", None)
                if sender_id in (None, "", 0, "0"):
                    sender_id = getattr(component, "qq", None)
            kind = kind.rsplit(".", 1)[-1]
            class_name = component.__class__.__name__.strip().lower()
            if kind not in {"reply", "quote"} and class_name not in {"reply", "quote"}:
                continue
            if sender_id is not None and str(sender_id).strip() == self_id:
                return True
        return False

    @staticmethod
    def _set_meta(event: AstrMessageEvent, value: dict[str, Any]):
        try:
            event.set_extra(META_ATTR, value)
        except Exception:
            pass

    @staticmethod
    def _get_meta(event: AstrMessageEvent) -> dict[str, Any]:
        try:
            value = event.get_extra(META_ATTR, {})
        except Exception:
            value = {}
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _group_id(event: AstrMessageEvent) -> str:
        message_obj = getattr(event, "message_obj", None)
        value = getattr(message_obj, "group_id", None)
        if value is None:
            getter = getattr(event, "get_group_id", None)
            value = getter() if callable(getter) else None
        return str(value).strip() if value is not None else ""

    @staticmethod
    def _sender_id(event: AstrMessageEvent) -> str:
        getter = getattr(event, "get_sender_id", None)
        if callable(getter):
            value = getter()
            if value is not None:
                return str(value).strip()
        message_obj = getattr(event, "message_obj", None)
        sender = getattr(message_obj, "sender", None)
        value = getattr(sender, "user_id", None)
        return str(value).strip() if value is not None else ""

    @staticmethod
    def _session(event: AstrMessageEvent) -> str:
        group_id = SmartReplyGatePlugin._group_id(event)
        if group_id:
            return f"群:{group_id}"
        return str(getattr(event, "unified_msg_origin", "私聊") or "私聊")

    @staticmethod
    def _get_data_dir() -> Path:
        try:
            from astrbot.core.utils.astrbot_path import get_astrbot_data_path

            return Path(get_astrbot_data_path()) / "plugin_data" / PLUGIN_NAME
        except Exception:
            return Path(__file__).resolve().parent / "data"

    def _bool_config(self, name: str, default: bool) -> bool:
        value = self.config.get(name, default)
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "on"}
        return bool(value)

    def _int_config(self, name: str, default: int) -> int:
        try:
            return int(self.config.get(name, default))
        except (TypeError, ValueError):
            return default

    def _float_config(self, name: str, default: float) -> float:
        try:
            return float(self.config.get(name, default))
        except (TypeError, ValueError):
            return default
