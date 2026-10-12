from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List
from unittest.mock import AsyncMock, Mock

import asyncio
import json
import pytest
import sys


PLUGIN_DIR = Path(__file__).resolve().parents[1]
MODULE_NAME = "report_to_admin_plugin_under_test"
SPEC = spec_from_file_location(MODULE_NAME, PLUGIN_DIR / "plugin.py")
assert SPEC is not None and SPEC.loader is not None
plugin_module = module_from_spec(SPEC)
sys.modules[MODULE_NAME] = plugin_module
SPEC.loader.exec_module(plugin_module)

ADMIN_QQ = "998877665"
ADMIN_GROUP = "776655443"
BOT_QQ = "123456789"
SOURCE_STREAM = "source-group"
ADMIN_STREAM = "admin-private"
GROUP_STREAM = "admin-group"


def make_message(message_id: str, *, session_id: str = SOURCE_STREAM, timestamp: str = "100") -> Dict[str, Any]:
    return {
        "message_id": message_id,
        "timestamp": timestamp,
        "platform": "qq",
        "account_id": BOT_QQ,
        "scope": "qq-default",
        "session_id": session_id,
        "message_info": {
            "user_info": {"user_id": "22334455", "user_nickname": "Target"},
            "group_info": {"group_id": "66778899", "group_name": "Target group"},
        },
        "raw_message": [
            {"type": "text", "data": f"Original evidence {message_id}"},
            {"type": "image", "data": "image-description", "binary_data_base64": "aW1hZ2U="},
        ],
    }


@pytest.fixture
def messages() -> Dict[str, Dict[str, Any]]:
    return {message_id: make_message(message_id) for message_id in ["m1", "m2", "m3"]}


@pytest.fixture
def ctx(messages: Dict[str, Dict[str, Any]]) -> Iterator[SimpleNamespace]:
    async def get_by_id(message_id: str, **kwargs: Any) -> Any:
        del kwargs
        return deepcopy(messages.get(message_id))

    async def trigger(stream_id: str, intent: str, **kwargs: Any) -> Dict[str, Any]:
        del intent, kwargs
        return {"success": True, "queued": True, "task_id": f"task-{stream_id}"}

    context = SimpleNamespace(
        logger=Mock(),
        message=SimpleNamespace(get_by_id=AsyncMock(side_effect=get_by_id)),
        chat=SimpleNamespace(open_session=AsyncMock(return_value={"success": True, "stream_id": ADMIN_STREAM})),
        send=SimpleNamespace(
            forward=AsyncMock(return_value={"sent": True, "message_id": "forward-id"}),
            text=AsyncMock(return_value={"sent": True, "message_id": "notice-id"}),
        ),
        maisaka=SimpleNamespace(
            context=SimpleNamespace(append=AsyncMock(return_value={"success": True})),
            proactive=SimpleNamespace(
                trigger=AsyncMock(
                    return_value={"success": True, "queued": True, "task_id": "task-id"}, side_effect=trigger
                )
            ),
        ),
    )
    yield context
    context.send.text.assert_not_awaited()


@pytest.fixture
def now(monkeypatch: pytest.MonkeyPatch) -> List[float]:
    clock = [10000.0]
    # Replace the plugin's reference, not time.monotonic used by the asyncio loop.
    monkeypatch.setattr(plugin_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    return clock


@pytest.fixture
def plugin(ctx: SimpleNamespace, now: List[float]) -> Any:
    del now
    instance = plugin_module.create_plugin()
    instance._set_context(ctx)
    instance.set_plugin_config(
        {"plugin": {"enabled": True, "config_version": "1.2.0"}, "report": {"admin_qq": ADMIN_QQ}}
    )
    return instance


@pytest.fixture
def dual_plugin(plugin: Any, ctx: SimpleNamespace) -> Any:
    plugin.config.report.group_enabled = True
    plugin.config.report.admin_group_id = ADMIN_GROUP

    async def open_session(**kwargs: Any) -> Dict[str, Any]:
        target_stream = ADMIN_STREAM if kwargs["chat_type"] == "private" else GROUP_STREAM
        return {"success": True, "stream_id": target_stream}

    ctx.chat.open_session.side_effect = open_session
    return plugin


def assert_no_delivery(ctx: SimpleNamespace) -> None:
    ctx.chat.open_session.assert_not_awaited()
    ctx.send.forward.assert_not_awaited()
    ctx.send.text.assert_not_awaited()
    ctx.maisaka.context.append.assert_not_awaited()
    ctx.maisaka.proactive.trigger.assert_not_awaited()


def delivery_by_type(result: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {delivery["chat_type"]: delivery for delivery in result["deliveries"]}


def destination_call_count(mock: AsyncMock, stream_id: str, stream_arg: int = 1) -> int:
    return sum(call.args[stream_arg] == stream_id for call in mock.await_args_list)


def test_sdk_declares_only_planner_tool_and_minimal_capabilities() -> None:
    plugin = plugin_module.create_plugin()
    components = plugin.get_components()
    assert len(components) == 1
    assert components[0]["name"] == "report_to_admin"
    assert components[0]["type"] == "TOOL"
    assert plugin.get_llm_providers() == []
    parameters = components[0]["metadata"]["parameters_raw"]
    assert set(parameters["properties"]) == {"message_ids", "reply_reference"}
    assert set(parameters["required"]) == {"message_ids", "reply_reference"}
    assert parameters["properties"]["reply_reference"]["maxLength"] == 2000
    assert parameters["additionalProperties"] is False
    manifest = json.loads((PLUGIN_DIR / "_manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["capabilities"]) == {
        "message.get_by_id",
        "chat.open_session",
        "send.forward",
        "maisaka.context.append",
        "maisaka.proactive.trigger",
    }


def test_sdk_generates_disabled_config_without_embedded_admin() -> None:
    config = plugin_module.create_plugin().get_default_config()
    assert config["plugin"] == {"enabled": False, "config_version": "1.2.0"}
    assert config["report"] == {
        "admin_qq": "",
        "private_enabled": True,
        "group_enabled": False,
        "admin_group_id": "",
        "cooldown_seconds": 60,
        "max_reports_per_hour": 20,
        "dedup_seconds": 86400,
    }


@pytest.mark.asyncio
async def test_success_preserves_original_evidence_and_account_route(
    plugin: Any, ctx: SimpleNamespace, messages: Dict[str, Dict[str, Any]]
) -> None:
    messages["m1"]["timestamp"] = "200"
    messages["m2"]["timestamp"] = "100"
    stages = Mock()
    stages.attach_mock(ctx.chat.open_session, "open")
    stages.attach_mock(ctx.maisaka.context.append, "append")
    stages.attach_mock(ctx.send.forward, "forward")
    stages.attach_mock(ctx.maisaka.proactive.trigger, "trigger")
    result = await plugin.report_to_admin(["m1", "m2"], "Repeated harassment", stream_id=SOURCE_STREAM)
    assert result["success"] is True
    assert result["status"] == "queued"
    assert result["stop_after_execution"] is True
    assert ctx.message.get_by_id.await_count == 2
    for message_id in ["m1", "m2"]:
        ctx.message.get_by_id.assert_any_await(message_id, stream_id=SOURCE_STREAM, include_binary_data=True)
    ctx.chat.open_session.assert_awaited_once_with(
        platform="qq", chat_type="private", user_id=ADMIN_QQ, account_id=BOT_QQ, scope="qq-default"
    )
    nodes, destination = ctx.send.forward.await_args.args
    assert destination == ADMIN_STREAM
    assert [node["message_id"] for node in nodes] == ["m2", "m1"]
    for node in nodes:
        assert node["segments"] == messages[node["message_id"]]["raw_message"]
        assert node["user_id"] == "22334455"
        assert node["nickname"] == "Target"
    assert ctx.send.forward.await_args.kwargs == {
        "return_details": True,
        "sync_to_maisaka_history": True,
        "maisaka_source_kind": "plugin:report_evidence",
    }
    destination, segments = ctx.maisaka.context.append.await_args.args
    assert destination == ADMIN_STREAM
    background = ctx.maisaka.context.append.await_args.kwargs["visible_text"]
    assert segments == [{"type": "text", "content": background}]
    assert "Repeated harassment" in background
    assert "Target group" in background
    assert "66778899" in background
    assert "22334455" in background
    assert ctx.maisaka.context.append.await_args.kwargs["source_kind"] == "plugin:report_background"
    destination, intent = ctx.maisaka.proactive.trigger.await_args.args
    assert destination == ADMIN_STREAM
    assert "forward-id" in intent
    assert "正常 reply" in intent
    assert "set_quote=false" in intent
    assert ctx.maisaka.proactive.trigger.await_args.kwargs["metadata"] == {
        "source_stream_id": SOURCE_STREAM,
        "source_message_ids": ["m1", "m2"],
        "forward_message_id": "forward-id",
        "chat_type": "private",
        "target_id": ADMIN_QQ,
    }
    assert result["deliveries"][0]["task_id"] == f"task-{ADMIN_STREAM}"
    assert result["deliveries"][0]["report_queued"] is True
    assert "尚不代表报告已送达" in result["content"]
    assert [call[0] for call in stages.mock_calls] == ["open", "append", "forward", "trigger"]


@pytest.mark.asyncio
async def test_private_source_is_identified_without_group_metadata(
    plugin: Any, ctx: SimpleNamespace, messages: Dict[str, Dict[str, Any]]
) -> None:
    messages["m1"]["message_info"]["group_info"] = None
    result = await plugin.report_to_admin(["m1"], "Harassment", stream_id=SOURCE_STREAM)
    assert result["success"] is True
    background = ctx.maisaka.context.append.await_args.kwargs["visible_text"]
    assert "私聊：Target（22334455）" in background


@pytest.mark.asyncio
async def test_disabled_plugin_does_not_query_or_send(plugin: Any, ctx: SimpleNamespace) -> None:
    plugin.config.plugin.enabled = False
    result = await plugin.report_to_admin(["m1"], "Harassment", stream_id=SOURCE_STREAM)
    assert result["success"] is False
    ctx.message.get_by_id.assert_not_awaited()
    assert_no_delivery(ctx)


@pytest.mark.asyncio
@pytest.mark.parametrize("admin_qq", ["", "1234", "1234567890123", "not-a-qq", "９９８８７７６６５"])
async def test_enabled_plugin_rejects_invalid_admin(plugin: Any, ctx: SimpleNamespace, admin_qq: str) -> None:
    plugin.config.report.admin_qq = admin_qq
    with pytest.raises(ValueError, match="admin_qq"):
        await plugin.on_load()
    assert_no_delivery(ctx)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message_ids", "reply_reference", "stream_id", "kwargs"),
    [
        ([], "Reason", SOURCE_STREAM, {}),
        (["m1"] * 6, "Reason", SOURCE_STREAM, {}),
        ("m1", "Reason", SOURCE_STREAM, {}),
        ([123], "Reason", SOURCE_STREAM, {}),
        ([" "], "Reason", SOURCE_STREAM, {}),
        (["m1"], " ", SOURCE_STREAM, {}),
        (["m1"], "x" * 2001, SOURCE_STREAM, {}),
        (["m1"], None, SOURCE_STREAM, {}),
        (["m1"], "Reason", "", {}),
        (["m1"], "Reason", SOURCE_STREAM, {"chat_id": "other-stream"}),
    ],
)
async def test_invalid_tool_arguments_never_query_or_send(
    plugin: Any, ctx: SimpleNamespace, message_ids: Any, reply_reference: Any, stream_id: str, kwargs: Dict[str, Any]
) -> None:
    result = await plugin.report_to_admin(message_ids, reply_reference, stream_id=stream_id, **kwargs)
    assert result["success"] is False
    ctx.message.get_by_id.assert_not_awaited()
    assert_no_delivery(ctx)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("session_id", "other-stream"),
        ("platform", "discord"),
        ("message_id", "other-id"),
        ("raw_message", []),
        ("raw_message", "fabricated text"),
    ],
)
async def test_foreign_or_invalid_evidence_is_not_sent(
    plugin: Any, ctx: SimpleNamespace, messages: Dict[str, Dict[str, Any]], field: str, value: Any
) -> None:
    messages["m1"][field] = value
    result = await plugin.report_to_admin(["m1"], "Reason", stream_id=SOURCE_STREAM)
    assert result["success"] is False
    assert_no_delivery(ctx)


@pytest.mark.asyncio
@pytest.mark.parametrize("media_type", ["image", "emoji", "voice"])
@pytest.mark.parametrize("binary_data", [None, ""])
async def test_media_without_original_binary_is_not_sent(
    plugin: Any,
    ctx: SimpleNamespace,
    messages: Dict[str, Dict[str, Any]],
    media_type: str,
    binary_data: Any,
) -> None:
    segment = {"type": media_type, "data": "Description is not evidence", "hash": "retained-hash"}
    if binary_data is not None:
        segment["binary_data_base64"] = binary_data
    messages["m1"]["raw_message"] = [segment]
    result = await plugin.report_to_admin(["m1"], "Reason", stream_id=SOURCE_STREAM)
    assert result["success"] is False
    assert_no_delivery(ctx)


@pytest.mark.asyncio
@pytest.mark.parametrize("media_available", [False, True])
async def test_nested_forward_requires_original_media_at_every_depth(
    plugin: Any, ctx: SimpleNamespace, messages: Dict[str, Dict[str, Any]], media_available: bool
) -> None:
    media = {"type": "image", "data": "Nested image description", "hash": "retained-hash"}
    if media_available:
        media["binary_data_base64"] = "aW1hZ2U="
    segments = [
        {
            "type": "forward",
            "data": [
                {
                    "user_id": "44556677",
                    "nickname": "Quoted sender",
                    "content": [
                        {"type": "text", "data": "Quoted evidence"},
                        {
                            "type": "forward",
                            "data": [{"user_id": "88990011", "nickname": "Nested sender", "content": [media]}],
                        },
                    ],
                }
            ],
        }
    ]
    messages["m1"]["raw_message"] = segments
    result = await plugin.report_to_admin(["m1"], "Reason", stream_id=SOURCE_STREAM)
    assert result["success"] is media_available
    if media_available:
        ctx.send.forward.assert_awaited_once()
        nodes, destination = ctx.send.forward.await_args.args
        assert destination == ADMIN_STREAM
        assert nodes[0]["segments"] == segments
        ctx.maisaka.context.append.assert_awaited_once()
        ctx.maisaka.proactive.trigger.assert_awaited_once()
    else:
        assert_no_delivery(ctx)


@pytest.mark.asyncio
async def test_missing_message_is_not_sent(plugin: Any, ctx: SimpleNamespace) -> None:
    result = await plugin.report_to_admin(["missing-id"], "Reason", stream_id=SOURCE_STREAM)
    assert result["success"] is False
    assert_no_delivery(ctx)


@pytest.mark.asyncio
@pytest.mark.parametrize("user_id", [ADMIN_QQ, BOT_QQ])
async def test_admin_and_bot_messages_are_not_reportable(
    plugin: Any, ctx: SimpleNamespace, messages: Dict[str, Dict[str, Any]], user_id: str
) -> None:
    messages["m1"]["message_info"]["user_info"]["user_id"] = user_id
    result = await plugin.report_to_admin(["m1"], "Reason", stream_id=SOURCE_STREAM)
    assert result["success"] is False
    assert_no_delivery(ctx)


@pytest.mark.asyncio
@pytest.mark.parametrize(("field", "value"), [("account_id", "another-bot"), ("scope", "another-scope")])
async def test_mixed_account_or_scope_is_not_sent(
    plugin: Any, ctx: SimpleNamespace, messages: Dict[str, Dict[str, Any]], field: str, value: str
) -> None:
    messages["m2"][field] = value
    result = await plugin.report_to_admin(["m1", "m2"], "Reason", stream_id=SOURCE_STREAM)
    assert result["success"] is False
    assert_no_delivery(ctx)


@pytest.mark.asyncio
async def test_bad_metadata_fails_before_forwarding(
    plugin: Any, ctx: SimpleNamespace, messages: Dict[str, Dict[str, Any]]
) -> None:
    del messages["m1"]["message_info"]["group_info"]["group_name"]
    with pytest.raises(KeyError):
        await plugin.report_to_admin(["m1"], "Reason", stream_id=SOURCE_STREAM)
    ctx.send.forward.assert_not_awaited()
    ctx.send.text.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("opened", [{"success": False, "error": "Unavailable"}, {"success": True, "stream_id": ""}])
async def test_admin_session_failure_is_not_success(plugin: Any, ctx: SimpleNamespace, opened: Dict[str, Any]) -> None:
    ctx.chat.open_session.return_value = opened
    result = await plugin.report_to_admin(["m1"], "Reason", stream_id=SOURCE_STREAM)
    assert result["success"] is False
    ctx.send.forward.assert_not_awaited()
    ctx.send.text.assert_not_awaited()


@pytest.mark.asyncio
async def test_forward_failure_does_not_queue_report_and_can_retry(plugin: Any, ctx: SimpleNamespace) -> None:
    ctx.send.forward.side_effect = [{"sent": False, "message_id": None}, {"sent": True, "message_id": "forward-id"}]
    result = await plugin.report_to_admin(["m1"], "Reason", stream_id=SOURCE_STREAM)
    assert result["success"] is False
    ctx.maisaka.proactive.trigger.assert_not_awaited()
    result = await plugin.report_to_admin(["m1"], "Reason", stream_id=SOURCE_STREAM)
    assert result["success"] is True
    assert ctx.send.forward.await_count == 2
    ctx.maisaka.context.append.assert_awaited_once()
    ctx.maisaka.proactive.trigger.assert_awaited_once()


@pytest.mark.asyncio
async def test_forward_exception_is_exposed_and_does_not_queue_report(plugin: Any, ctx: SimpleNamespace) -> None:
    ctx.send.forward.side_effect = RuntimeError("Adapter unavailable")
    failed = await plugin.report_to_admin(["m1"], "Prepared report", stream_id=SOURCE_STREAM)
    assert failed["success"] is False
    assert "RuntimeError: Adapter unavailable" in failed["deliveries"][0]["error"]
    assert failed["deliveries"][0]["evidence_forwarded"] is False
    ctx.logger.exception.assert_called_once()
    ctx.maisaka.proactive.trigger.assert_not_awaited()
    ctx.send.forward.side_effect = None
    result = await plugin.report_to_admin(["m1"], "Reason", stream_id=SOURCE_STREAM)
    assert result["success"] is True
    assert ctx.send.forward.await_count == 2


@pytest.mark.asyncio
async def test_trigger_failure_retries_only_queue_request_even_during_rate_limit(
    plugin: Any, ctx: SimpleNamespace
) -> None:
    plugin.config.report.max_reports_per_hour = 1
    ctx.maisaka.proactive.trigger.side_effect = [
        {"success": False, "queued": False},
        {"success": True, "queued": True, "task_id": "report-task"},
    ]
    first = await plugin.report_to_admin(["m1"], "Original reference", stream_id=SOURCE_STREAM)
    assert first["success"] is False
    assert first["evidence_forwarded"] is True
    second = await plugin.report_to_admin(["m1"], "Changed reference", stream_id=SOURCE_STREAM)
    assert second["success"] is True
    ctx.send.forward.assert_awaited_once()
    ctx.chat.open_session.assert_awaited_once()
    ctx.message.get_by_id.assert_awaited_once()
    ctx.maisaka.context.append.assert_awaited_once()
    assert ctx.maisaka.proactive.trigger.await_count == 2
    assert ctx.maisaka.proactive.trigger.await_args_list[0] == ctx.maisaka.proactive.trigger.await_args_list[1]
    background = ctx.maisaka.context.append.await_args.kwargs["visible_text"]
    assert "Original reference" in background
    assert "Changed reference" not in background
    assert second["deliveries"][0]["task_id"] == "report-task"


@pytest.mark.asyncio
async def test_trigger_exception_preserves_forward_record_for_retry(plugin: Any, ctx: SimpleNamespace) -> None:
    ctx.maisaka.proactive.trigger.side_effect = RuntimeError("Queue unavailable")
    failed = await plugin.report_to_admin(["m1"], "Prepared report", stream_id=SOURCE_STREAM)
    assert failed["success"] is False
    assert "RuntimeError: Queue unavailable" in failed["deliveries"][0]["error"]
    assert failed["deliveries"][0]["evidence_forwarded"] is True
    ctx.logger.exception.assert_called_once()
    ctx.maisaka.proactive.trigger.side_effect = None
    result = await plugin.report_to_admin(["m1"], "Reason", stream_id=SOURCE_STREAM)
    assert result["success"] is True
    ctx.send.forward.assert_awaited_once()
    ctx.maisaka.context.append.assert_awaited_once()
    assert ctx.maisaka.proactive.trigger.await_count == 2


@pytest.mark.asyncio
async def test_exact_duplicate_set_is_not_delivered_again(plugin: Any, ctx: SimpleNamespace) -> None:
    first = await plugin.report_to_admin(["m1", "m2", "m1"], "Reason", stream_id=SOURCE_STREAM)
    second = await plugin.report_to_admin(["m2", " m1 "], "Reason", stream_id=SOURCE_STREAM)
    assert first["success"] is True
    assert second["success"] is True
    assert second["status"] == "already_queued"
    assert second["stop_after_execution"] is True
    assert ctx.message.get_by_id.await_count == 2
    ctx.send.forward.assert_awaited_once()
    ctx.maisaka.context.append.assert_awaited_once()
    ctx.maisaka.proactive.trigger.assert_awaited_once()


@pytest.mark.asyncio
async def test_partial_duplicate_set_is_rejected(plugin: Any, ctx: SimpleNamespace) -> None:
    await plugin.report_to_admin(["m1", "m2"], "Reason", stream_id=SOURCE_STREAM)
    result = await plugin.report_to_admin(["m2", "m3"], "Reason", stream_id=SOURCE_STREAM)
    assert result["success"] is False
    assert result["duplicate_ids"] == ["m2"]
    ctx.send.forward.assert_awaited_once()
    ctx.maisaka.proactive.trigger.assert_awaited_once()
    assert ctx.message.get_by_id.await_count == 2


@pytest.mark.asyncio
async def test_concurrent_same_report_is_sent_once(plugin: Any, ctx: SimpleNamespace) -> None:
    async def forward(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        del args, kwargs
        await asyncio.sleep(0)
        return {"sent": True, "message_id": "forward-id"}

    ctx.send.forward.side_effect = forward
    results = await asyncio.gather(
        plugin.report_to_admin(["m1"], "Reason", stream_id=SOURCE_STREAM),
        plugin.report_to_admin(["m1"], "Reason", stream_id=SOURCE_STREAM),
    )
    assert all(result["success"] is True for result in results)
    assert {result["status"] for result in results} == {"queued", "already_queued"}
    ctx.send.forward.assert_awaited_once()
    ctx.maisaka.context.append.assert_awaited_once()
    ctx.maisaka.proactive.trigger.assert_awaited_once()


@pytest.mark.asyncio
async def test_source_cooldown_expires_at_boundary(plugin: Any, ctx: SimpleNamespace, now: List[float]) -> None:
    await plugin.report_to_admin(["m1"], "Reason", stream_id=SOURCE_STREAM)
    now[0] += 59
    denied = await plugin.report_to_admin(["m2"], "Reason", stream_id=SOURCE_STREAM)
    assert denied["success"] is False
    ctx.send.forward.assert_awaited_once()
    now[0] += 1
    accepted = await plugin.report_to_admin(["m2"], "Reason", stream_id=SOURCE_STREAM)
    assert accepted["success"] is True
    assert ctx.send.forward.await_count == 2


@pytest.mark.asyncio
async def test_hourly_limit_applies_across_sources_and_expires(
    plugin: Any, ctx: SimpleNamespace, messages: Dict[str, Dict[str, Any]], now: List[float]
) -> None:
    plugin.config.report.max_reports_per_hour = 1
    messages["m2"]["session_id"] = "other-source"
    await plugin.report_to_admin(["m1"], "Reason", stream_id=SOURCE_STREAM)
    denied = await plugin.report_to_admin(["m2"], "Reason", stream_id="other-source")
    assert denied["success"] is False
    ctx.send.forward.assert_awaited_once()
    now[0] += 3600
    accepted = await plugin.report_to_admin(["m2"], "Reason", stream_id="other-source")
    assert accepted["success"] is True
    assert ctx.send.forward.await_count == 2


@pytest.mark.asyncio
async def test_dedup_record_expires_at_configured_boundary(plugin: Any, ctx: SimpleNamespace, now: List[float]) -> None:
    plugin.config.report.dedup_seconds = 60
    plugin.config.report.cooldown_seconds = 0
    await plugin.report_to_admin(["m1"], "Reason", stream_id=SOURCE_STREAM)
    now[0] += 59
    duplicate = await plugin.report_to_admin(["m1"], "Reason", stream_id=SOURCE_STREAM)
    assert duplicate["status"] == "already_queued"
    ctx.send.forward.assert_awaited_once()
    now[0] += 1
    fresh = await plugin.report_to_admin(["m1"], "Reason", stream_id=SOURCE_STREAM)
    assert fresh["status"] == "queued"
    assert ctx.send.forward.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("admin_group_id", ["", "1234", "not-a-group", "７７６６５５４４３"])
async def test_enabled_group_delivery_rejects_invalid_group(
    dual_plugin: Any, ctx: SimpleNamespace, admin_group_id: str
) -> None:
    dual_plugin.config.report.admin_group_id = admin_group_id
    with pytest.raises(ValueError, match="admin_group_id"):
        await dual_plugin.on_load()
    assert_no_delivery(ctx)


@pytest.mark.asyncio
async def test_enabled_plugin_requires_at_least_one_target(plugin: Any, ctx: SimpleNamespace) -> None:
    plugin.config.report.private_enabled = False
    plugin.config.report.group_enabled = False
    with pytest.raises(ValueError, match="至少开启"):
        await plugin.on_load()
    assert_no_delivery(ctx)


@pytest.mark.asyncio
async def test_group_only_delivery_does_not_require_private_admin(dual_plugin: Any, ctx: SimpleNamespace) -> None:
    dual_plugin.config.report.private_enabled = False
    dual_plugin.config.report.admin_qq = ""
    await dual_plugin.on_load()
    result = await dual_plugin.report_to_admin(
        ["m1"], reply_reference="Group harassment facts", stream_id=SOURCE_STREAM
    )
    assert result["success"] is True
    assert set(delivery_by_type(result)) == {"group"}
    ctx.chat.open_session.assert_awaited_once_with(
        platform="qq", chat_type="group", group_id=ADMIN_GROUP, account_id=BOT_QQ, scope="qq-default"
    )
    assert ctx.send.forward.await_args.args[1] == GROUP_STREAM
    assert ctx.maisaka.context.append.await_args.args[0] == GROUP_STREAM
    assert ctx.maisaka.proactive.trigger.await_args.args[0] == GROUP_STREAM


@pytest.mark.asyncio
@pytest.mark.parametrize("length", [1001, 2000])
async def test_reply_reference_accepts_length_boundary(plugin: Any, ctx: SimpleNamespace, length: int) -> None:
    reply_reference = "x" * length
    result = await plugin.report_to_admin(["m1"], reply_reference=reply_reference, stream_id=SOURCE_STREAM)
    assert result["success"] is True
    assert reply_reference in ctx.maisaka.context.append.await_args.kwargs["visible_text"]


@pytest.mark.asyncio
async def test_dual_delivery_uses_same_evidence_background_and_separate_normal_reply_tasks(
    dual_plugin: Any, ctx: SimpleNamespace
) -> None:
    async def forward(nodes: Any, stream_id: str, **kwargs: Any) -> Dict[str, Any]:
        del nodes, kwargs
        return {"sent": True, "message_id": f"forward-{stream_id}"}

    ctx.send.forward.side_effect = forward
    reply_reference = "对方持续骚扰并连续发布威胁。\n请向管理员说明相关事实，请求核实原消息。"
    result = await dual_plugin.report_to_admin(["m1"], reply_reference=reply_reference, stream_id=SOURCE_STREAM)
    assert result["success"] is True
    assert result["status"] == "queued"
    assert result["stop_after_execution"] is True
    deliveries = delivery_by_type(result)
    assert set(deliveries) == {"private", "group"}
    assert deliveries["private"]["target_id"] == ADMIN_QQ
    assert deliveries["group"]["target_id"] == ADMIN_GROUP
    assert all(item["evidence_forwarded"] and item["report_queued"] for item in deliveries.values())
    assert deliveries["private"]["task_id"] == f"task-{ADMIN_STREAM}"
    assert deliveries["group"]["task_id"] == f"task-{GROUP_STREAM}"
    ctx.message.get_by_id.assert_awaited_once_with("m1", stream_id=SOURCE_STREAM, include_binary_data=True)
    ctx.chat.open_session.assert_any_await(
        platform="qq", chat_type="private", user_id=ADMIN_QQ, account_id=BOT_QQ, scope="qq-default"
    )
    ctx.chat.open_session.assert_any_await(
        platform="qq", chat_type="group", group_id=ADMIN_GROUP, account_id=BOT_QQ, scope="qq-default"
    )
    assert ctx.send.forward.await_count == 2
    assert {call.args[1] for call in ctx.send.forward.await_args_list} == {ADMIN_STREAM, GROUP_STREAM}
    assert ctx.send.forward.await_args_list[0].args[0] is ctx.send.forward.await_args_list[1].args[0]
    assert ctx.maisaka.context.append.await_count == 2
    backgrounds = [call.kwargs["visible_text"] for call in ctx.maisaka.context.append.await_args_list]
    assert backgrounds[0] == backgrounds[1]
    assert reply_reference in backgrounds[0]
    assert ctx.maisaka.proactive.trigger.await_count == 2
    for call in ctx.maisaka.proactive.trigger.await_args_list:
        stream_id, intent = call.args
        assert call.kwargs["metadata"]["forward_message_id"] == f"forward-{stream_id}"
        assert f"forward-{stream_id}" in intent
        assert "正常 reply" in intent
        assert "set_quote=false" in intent
        recipient = "当前私聊中的管理员" if stream_id == ADMIN_STREAM else "当前管理群中的管理员"
        assert recipient in intent


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_type", ["private", "group"])
@pytest.mark.parametrize("stage", ["open_session", "append", "forward", "trigger"])
@pytest.mark.parametrize("failure_kind", ["false", "exception"])
async def test_one_target_failure_does_not_block_other_target_and_retry_only_completes_failure(
    dual_plugin: Any, ctx: SimpleNamespace, failed_type: str, stage: str, failure_kind: str
) -> None:
    dual_plugin.config.report.max_reports_per_hour = 1
    failed_stream = ADMIN_STREAM if failed_type == "private" else GROUP_STREAM
    successful_type = "group" if failed_type == "private" else "private"
    successful_stream = GROUP_STREAM if failed_type == "private" else ADMIN_STREAM
    permit_failed_target = [False]

    def failure() -> Dict[str, Any]:
        if failure_kind == "exception":
            raise RuntimeError(f"{stage} unavailable")
        return {"success": False, "sent": False, "message_id": None}

    async def open_session(**kwargs: Any) -> Dict[str, Any]:
        if kwargs["chat_type"] == failed_type and stage == "open_session" and not permit_failed_target[0]:
            return failure()
        target_stream = ADMIN_STREAM if kwargs["chat_type"] == "private" else GROUP_STREAM
        return {"success": True, "stream_id": target_stream}

    async def forward(nodes: Any, stream_id: str, **kwargs: Any) -> Dict[str, Any]:
        del nodes, kwargs
        if stream_id == failed_stream and stage == "forward" and not permit_failed_target[0]:
            return failure()
        return {"sent": True, "message_id": "forward-id"}

    async def append(stream_id: str, segments: Any, **kwargs: Any) -> Dict[str, Any]:
        del segments, kwargs
        if stream_id == failed_stream and stage == "append" and not permit_failed_target[0]:
            return failure()
        return {"success": True}

    async def trigger(stream_id: str, intent: str, **kwargs: Any) -> Dict[str, Any]:
        del intent, kwargs
        if stream_id == failed_stream and stage == "trigger" and not permit_failed_target[0]:
            return failure()
        return {"success": True, "queued": True, "task_id": f"task-{stream_id}"}

    ctx.chat.open_session.side_effect = open_session
    ctx.send.forward.side_effect = forward
    ctx.maisaka.context.append.side_effect = append
    ctx.maisaka.proactive.trigger.side_effect = trigger
    reply_reference = "Original facts\nPlease ask the administrator to review the evidence."
    first = await dual_plugin.report_to_admin(["m1"], reply_reference, stream_id=SOURCE_STREAM)
    assert first["success"] is False
    deliveries = delivery_by_type(first)
    assert deliveries[failed_type]["success"] is False
    assert deliveries[failed_type]["evidence_forwarded"] is (stage == "trigger")
    assert deliveries[failed_type]["report_queued"] is False
    assert deliveries[failed_type]["task_id"] is None
    assert deliveries[successful_type]["success"] is True
    assert deliveries[successful_type]["report_queued"] is True
    if failure_kind == "exception":
        assert f"RuntimeError: {stage} unavailable" in deliveries[failed_type]["error"]
        ctx.logger.exception.assert_called_once()

    permit_failed_target[0] = True
    retried = await dual_plugin.report_to_admin(
        ["m1"], "Changed reference must not replace the original", stream_id=SOURCE_STREAM
    )
    assert retried["success"] is True
    retry_deliveries = delivery_by_type(retried)
    assert retry_deliveries[successful_type]["status"] == "already_queued"
    assert retry_deliveries[failed_type]["status"] == "queued"
    assert destination_call_count(ctx.send.forward, successful_stream) == 1
    assert destination_call_count(ctx.maisaka.context.append, successful_stream, 0) == 1
    assert destination_call_count(ctx.maisaka.proactive.trigger, successful_stream, 0) == 1
    assert destination_call_count(ctx.send.forward, failed_stream) == (2 if stage == "forward" else 1)
    assert destination_call_count(ctx.maisaka.context.append, failed_stream, 0) == (2 if stage == "append" else 1)
    assert destination_call_count(ctx.maisaka.proactive.trigger, failed_stream, 0) == (2 if stage == "trigger" else 1)
    assert ctx.message.get_by_id.await_count == (1 if stage == "trigger" else 2)
    assert ctx.chat.open_session.await_count == (3 if stage == "open_session" else 2)
    assert all(reply_reference in call.kwargs["visible_text"] for call in ctx.maisaka.context.append.await_args_list)


@pytest.mark.asyncio
async def test_two_destinations_consume_one_hourly_report_quota(dual_plugin: Any, ctx: SimpleNamespace) -> None:
    dual_plugin.config.report.cooldown_seconds = 0
    dual_plugin.config.report.max_reports_per_hour = 2
    first = await dual_plugin.report_to_admin(["m1"], "First prepared report", stream_id=SOURCE_STREAM)
    second = await dual_plugin.report_to_admin(["m2"], "Second prepared report", stream_id=SOURCE_STREAM)
    assert first["success"] is True
    assert second["success"] is True
    assert ctx.send.forward.await_count == 4
    denied = await dual_plugin.report_to_admin(["m3"], "Third prepared report", stream_id=SOURCE_STREAM)
    assert denied["success"] is False
    assert ctx.send.forward.await_count == 4


@pytest.mark.asyncio
async def test_zero_forward_record_preserves_reference_without_consuming_quota_or_bypassing_other_report_limits(
    plugin: Any, ctx: SimpleNamespace, now: List[float]
) -> None:
    plugin.config.report.max_reports_per_hour = 1
    ctx.send.forward.side_effect = [
        {"sent": False, "message_id": None},
        {"sent": True, "message_id": "other-forward"},
        {"sent": True, "message_id": "retried-forward"},
    ]
    failed = await plugin.report_to_admin(["m1"], "Original prepared report", stream_id=SOURCE_STREAM)
    assert failed["success"] is False
    assert failed["evidence_forwarded"] is False
    other = await plugin.report_to_admin(["m2"], "Other report", stream_id=SOURCE_STREAM)
    assert other["success"] is True
    denied_cooldown = await plugin.report_to_admin(["m1"], "Changed report", stream_id=SOURCE_STREAM)
    assert denied_cooldown["success"] is False
    assert ctx.send.forward.await_count == 2
    now[0] += 60
    denied_quota = await plugin.report_to_admin(["m1"], "Changed report", stream_id=SOURCE_STREAM)
    assert denied_quota["success"] is False
    assert ctx.send.forward.await_count == 2
    now[0] += 3540
    accepted = await plugin.report_to_admin(["m1"], "Changed report", stream_id=SOURCE_STREAM)
    assert accepted["success"] is True
    assert ctx.send.forward.await_count == 3
    first_background = ctx.maisaka.context.append.await_args_list[0].kwargs["visible_text"]
    assert "Original prepared report" in first_background
    assert "Changed report" not in first_background
    assert ctx.maisaka.context.append.await_count == 2
    assert ctx.maisaka.proactive.trigger.await_count == 2


@pytest.mark.asyncio
async def test_zero_forward_cached_set_does_not_block_other_selection_or_bypass_new_overlap(
    plugin: Any, ctx: SimpleNamespace
) -> None:
    plugin.config.report.cooldown_seconds = 0
    ctx.send.forward.side_effect = [{"sent": False, "message_id": None}, {"sent": True, "message_id": "forward-id"}]
    failed = await plugin.report_to_admin(["m1", "m2"], "Original prepared report", stream_id=SOURCE_STREAM)
    assert failed["success"] is False
    other = await plugin.report_to_admin(["m1"], "Alternative selection report", stream_id=SOURCE_STREAM)
    assert other["success"] is True
    retried = await plugin.report_to_admin(["m1", "m2"], "Changed report", stream_id=SOURCE_STREAM)
    assert retried["success"] is False
    assert retried["duplicate_ids"] == ["m1"]
    assert ctx.send.forward.await_count == 2
    assert ctx.message.get_by_id.await_count == 3


@pytest.mark.asyncio
async def test_invalid_evidence_on_retry_does_not_block_pending_task_for_forwarded_target(
    dual_plugin: Any, ctx: SimpleNamespace, messages: Dict[str, Dict[str, Any]]
) -> None:
    permit_private_forward = [False]
    permit_trigger = [False]

    async def forward(nodes: Any, stream_id: str, **kwargs: Any) -> Dict[str, Any]:
        del nodes, kwargs
        sent = stream_id != ADMIN_STREAM or permit_private_forward[0]
        return {"sent": sent, "message_id": "forward-id" if sent else None}

    async def trigger(stream_id: str, intent: str, **kwargs: Any) -> Dict[str, Any]:
        del intent, kwargs
        return {"success": permit_trigger[0], "queued": permit_trigger[0], "task_id": f"task-{stream_id}"}

    ctx.send.forward.side_effect = forward
    ctx.maisaka.proactive.trigger.side_effect = trigger
    failed = await dual_plugin.report_to_admin(["m1"], "Original prepared report", stream_id=SOURCE_STREAM)
    first_deliveries = delivery_by_type(failed)
    assert first_deliveries["private"]["evidence_forwarded"] is False
    assert first_deliveries["group"]["evidence_forwarded"] is True
    assert first_deliveries["group"]["report_queued"] is False

    original_segments = deepcopy(messages["m1"]["raw_message"])
    messages["m1"]["raw_message"] = []
    permit_trigger[0] = True
    retried = await dual_plugin.report_to_admin(["m1"], "Changed report", stream_id=SOURCE_STREAM)
    assert retried["success"] is False
    retry_deliveries = delivery_by_type(retried)
    assert retry_deliveries["private"]["success"] is False
    assert retry_deliveries["private"]["error"]
    assert retry_deliveries["group"]["success"] is True
    assert retry_deliveries["group"]["report_queued"] is True
    assert destination_call_count(ctx.send.forward, ADMIN_STREAM) == 1
    assert destination_call_count(ctx.send.forward, GROUP_STREAM) == 1
    assert destination_call_count(ctx.maisaka.proactive.trigger, GROUP_STREAM, 0) == 2

    messages["m1"]["raw_message"] = original_segments
    permit_private_forward[0] = True
    completed = await dual_plugin.report_to_admin(["m1"], "Changed report again", stream_id=SOURCE_STREAM)
    assert completed["success"] is True
    assert destination_call_count(ctx.send.forward, ADMIN_STREAM) == 2
    assert destination_call_count(ctx.send.forward, GROUP_STREAM) == 1
    assert destination_call_count(ctx.maisaka.proactive.trigger, GROUP_STREAM, 0) == 2
    assert all(
        "Original prepared report" in call.kwargs["visible_text"] for call in ctx.maisaka.context.append.await_args_list
    )


@pytest.mark.asyncio
async def test_adding_and_changing_target_reuses_reference_without_resending_or_requeueing_completed_target(
    dual_plugin: Any, ctx: SimpleNamespace
) -> None:
    dual_plugin.config.report.group_enabled = False
    first = await dual_plugin.report_to_admin(["m1"], "Original prepared report", stream_id=SOURCE_STREAM)
    assert first["success"] is True
    dual_plugin.config.report.group_enabled = True
    added = await dual_plugin.report_to_admin(["m1"], "Changed report", stream_id=SOURCE_STREAM)
    assert added["success"] is True
    assert destination_call_count(ctx.send.forward, ADMIN_STREAM) == 1
    assert destination_call_count(ctx.send.forward, GROUP_STREAM) == 1
    assert delivery_by_type(added)["private"]["status"] == "already_queued"

    dual_plugin.config.report.private_enabled = False
    dual_plugin.config.report.admin_group_id = "554433221"
    changed = await dual_plugin.report_to_admin(["m1"], "Changed report again", stream_id=SOURCE_STREAM)
    assert changed["success"] is True
    assert set(delivery_by_type(changed)) == {"group"}
    assert delivery_by_type(changed)["group"]["target_id"] == "554433221"
    assert destination_call_count(ctx.send.forward, ADMIN_STREAM) == 1
    assert destination_call_count(ctx.send.forward, GROUP_STREAM) == 2
    ctx.chat.open_session.assert_any_await(
        platform="qq", chat_type="group", group_id="554433221", account_id=BOT_QQ, scope="qq-default"
    )
    assert ctx.message.get_by_id.await_count == 3
    assert all(
        "Original prepared report" in call.kwargs["visible_text"] for call in ctx.maisaka.context.append.await_args_list
    )
    assert destination_call_count(ctx.maisaka.proactive.trigger, ADMIN_STREAM, 0) == 1
    assert destination_call_count(ctx.maisaka.proactive.trigger, GROUP_STREAM, 0) == 2


@pytest.mark.asyncio
async def test_removed_unfinished_target_is_not_retried(dual_plugin: Any, ctx: SimpleNamespace) -> None:
    async def forward(nodes: Any, stream_id: str, **kwargs: Any) -> Dict[str, Any]:
        del nodes, kwargs
        return {"sent": stream_id != GROUP_STREAM, "message_id": None}

    ctx.send.forward.side_effect = forward
    failed = await dual_plugin.report_to_admin(["m1"], "Original prepared report", stream_id=SOURCE_STREAM)
    assert failed["success"] is False
    dual_plugin.config.report.group_enabled = False
    retried = await dual_plugin.report_to_admin(["m1"], "Changed report", stream_id=SOURCE_STREAM)
    assert retried["success"] is True
    assert retried["status"] == "already_queued"
    assert set(delivery_by_type(retried)) == {"private"}
    assert ctx.send.forward.await_count == 2
    ctx.maisaka.proactive.trigger.assert_awaited_once()
    ctx.message.get_by_id.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "queue_result",
    [
        {"success": False, "queued": True, "task_id": "unconfirmed-task"},
        {"success": True, "queued": False, "task_id": "unconfirmed-task"},
        {"success": True, "task_id": "unconfirmed-task"},
        {"success": True, "queued": True},
        {"success": True, "queued": True, "task_id": ""},
        {"success": True, "queued": "true", "task_id": "unconfirmed-task"},
        {"success": True, "queued": 1, "task_id": "unconfirmed-task"},
    ],
)
async def test_incomplete_queue_confirmation_is_failure_and_retry_only_triggers_task(
    plugin: Any, ctx: SimpleNamespace, queue_result: Dict[str, Any]
) -> None:
    plugin.config.report.max_reports_per_hour = 1
    ctx.maisaka.proactive.trigger.side_effect = None
    ctx.maisaka.proactive.trigger.return_value = queue_result
    failed = await plugin.report_to_admin(["m1"], "Original facts", stream_id=SOURCE_STREAM)
    assert failed["success"] is False
    assert failed["evidence_forwarded"] is True
    assert failed["deliveries"][0]["report_queued"] is False
    assert failed["deliveries"][0]["task_id"] is None
    assert failed.get("stop_after_execution") is not True
    ctx.maisaka.proactive.trigger.return_value = {"success": True, "queued": True, "task_id": "confirmed-task"}
    retried = await plugin.report_to_admin(["m1"], "Changed facts", stream_id=SOURCE_STREAM)
    assert retried["success"] is True
    assert retried["status"] == "queued"
    assert retried["deliveries"][0]["task_id"] == "confirmed-task"
    ctx.message.get_by_id.assert_awaited_once()
    ctx.chat.open_session.assert_awaited_once()
    ctx.maisaka.context.append.assert_awaited_once()
    ctx.send.forward.assert_awaited_once()
    assert ctx.maisaka.proactive.trigger.await_count == 2


@pytest.mark.asyncio
async def test_missing_forward_message_id_requests_real_history_anchor_without_fabricating_id(
    plugin: Any, ctx: SimpleNamespace
) -> None:
    ctx.send.forward.return_value = {"sent": True, "message_id": None}
    result = await plugin.report_to_admin(["m1"], "Original facts", stream_id=SOURCE_STREAM)
    assert result["success"] is True
    assert result["status"] == "queued"
    trigger_call = ctx.maisaka.proactive.trigger.await_args
    assert trigger_call.kwargs["metadata"]["forward_message_id"] is None
    assert "请从当前会话历史选择你刚发送的真实证据转发消息编号" in trigger_call.args[1]
    assert f"task-{ADMIN_STREAM}" not in trigger_call.args[1]
    assert result["deliveries"][0]["task_id"] == f"task-{ADMIN_STREAM}"
