from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator
from typing import Any

from app.models.chat import ChatSettings
from app.services.llm import LLMClient, LlamaServerUnavailableError
from app.services.mcp_client import MCPClient
from app.services.retriever import Retriever

logger = logging.getLogger(__name__)

_MAX_TOOL_ITERATIONS = 10

# Matches Mistral's [TOOL_CALLS] token followed by a JSON array
_MISTRAL_TOOL_RE = re.compile(r"\[TOOL_CALLS\]\s*(\[.*?\])", re.DOTALL)


def _parse_mistral_tool_calls(content: str) -> tuple[list[dict[str, Any]], str]:
    """
    Detect and normalise a Mistral-style [TOOL_CALLS] block into the OpenAI
    tool_calls structure.  Returns (tool_calls, cleaned_content).
    """
    match = _MISTRAL_TOOL_RE.search(content)
    if not match:
        return [], content

    try:
        raw_calls: list[dict[str, Any]] = json.loads(match.group(1))
    except json.JSONDecodeError:
        logger.warning("Failed to parse Mistral tool calls JSON")
        return [], content

    tool_calls: list[dict[str, Any]] = []
    for i, call in enumerate(raw_calls):
        tool_calls.append(
            {
                "id": f"call_mistral_{i}",
                "type": "function",
                "function": {
                    "name": call.get("name", ""),
                    "arguments": json.dumps(call.get("arguments", {})),
                },
            }
        )

    cleaned = content[: match.start()].strip()
    return tool_calls, cleaned


def _content_item_to_str(item: Any) -> str:
    """Serialize a single MCP content item to a string for the tool result message."""
    if hasattr(item, "text"):
        return item.text
    if isinstance(item, dict):
        return item.get("text", json.dumps(item))
    return str(item)


def _serialize_tool_result(content: Any) -> str:
    if isinstance(content, list):
        parts = [_content_item_to_str(c) for c in content]
        return "\n".join(parts)
    return _content_item_to_str(content)


async def run_chat(
    history: list[dict[str, Any]],
    settings: ChatSettings,
    llm_client: LLMClient,
    mcp_client: MCPClient,
    retriever: Retriever | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """
    Core tool-call loop.  Yields SSE event dicts:
      {"type": "token",       "text": "..."}
      {"type": "tool_start",  "name": "...", "args": {...}}
      {"type": "tool_result", "name": "...", "result": "..."}
      {"type": "tool_error",  "name": "...", "error": "..."}
      {"type": "done"}
      {"type": "error",       "message": "..."}
    """
    tools = mcp_client.tools if mcp_client.available else []
    logger.info("run_chat: %d tool(s) available to model: %s",
                len(tools),
                [t["function"]["name"] for t in tools] or "(none)")

    # --- RAG: build a one-shot messages list with context appended to the
    #     system prompt. We never mutate the stored history. ----------------
    llm_messages = list(history)  # shallow copy — safe to modify positions

    if retriever and retriever.available and retriever.document_count > 0:
        user_msgs = [m for m in history if m.get("role") == "user"]
        if user_msgs:
            query_text = user_msgs[-1].get("content", "")
            chunks = retriever.query(query_text)
            if chunks:
                context_text = "\n\n---\n\n".join(chunks)
                rag_addition = f"\n\nRelevant context from knowledge base:\n\n{context_text}"

                if llm_messages and llm_messages[0].get("role") == "system":
                    # Append to existing system message (copy, don't modify history)
                    llm_messages[0] = {
                        **llm_messages[0],
                        "content": llm_messages[0]["content"] + rag_addition,
                    }
                else:
                    # No system message yet — prepend one
                    llm_messages.insert(0, {"role": "system", "content": rag_addition.strip()})

                logger.info("RAG: injected %d chunk(s) into context", len(chunks))

    for iteration in range(_MAX_TOOL_ITERATIONS):
        assistant_content = ""
        # tool_calls_acc is keyed by index
        tool_calls_acc: dict[int, dict[str, Any]] = {}

        try:
            async for chunk in llm_client.chat_stream(
                llm_messages,
                tools,
                temperature=settings.temperature,
                top_p=settings.top_p,
                max_tokens=settings.max_tokens,
            ):
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta", {})

                # --- content tokens -----------------------------------------------
                token_text = delta.get("content") or ""
                if token_text:
                    assistant_content += token_text
                    yield {"type": "token", "text": token_text}

                # --- tool call deltas (OpenAI streaming format) --------------------
                for tc_delta in delta.get("tool_calls") or []:
                    idx: int = tc_delta.get("index", 0)
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    entry = tool_calls_acc[idx]
                    if tc_delta.get("id"):
                        entry["id"] = tc_delta["id"]
                    func_delta = tc_delta.get("function") or {}
                    if func_delta.get("name"):
                        entry["function"]["name"] += func_delta["name"]
                    if func_delta.get("arguments"):
                        entry["function"]["arguments"] += func_delta["arguments"]

        except LlamaServerUnavailableError as exc:
            yield {"type": "error", "message": str(exc)}
            return

        # --- Post-stream: compile tool calls --------------------------------------
        tool_calls: list[dict[str, Any]] = [
            tool_calls_acc[i] for i in sorted(tool_calls_acc)
        ]

        logger.debug("Raw assistant_content after stream: %r", assistant_content[:500])
        logger.debug("OpenAI-format tool_calls_acc: %s", tool_calls_acc)

        # Fallback: detect Mistral [TOOL_CALLS] format
        if not tool_calls and "[TOOL_CALLS]" in assistant_content:
            tool_calls, assistant_content = _parse_mistral_tool_calls(assistant_content)

        # --- No tool calls → final response, we're done ---------------------------
        if not tool_calls:
            break

        # --- Dispatch tool calls --------------------------------------------------
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": assistant_content or None,
            "tool_calls": tool_calls,
        }
        history.append(assistant_msg)
        llm_messages.append(assistant_msg)

        for tc in tool_calls:
            name = tc["function"]["name"]
            try:
                args: dict[str, Any] = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}

            yield {"type": "tool_start", "name": name, "args": args}
            logger.info("Tool call → %s  args=%s", name, json.dumps(args, ensure_ascii=False)[:200])

            try:
                raw_result = await mcp_client.call_tool(name, args)
                result_str = _serialize_tool_result(raw_result)
                logger.info("Tool result ← %s  (%d chars)", name, len(result_str))
                yield {"type": "tool_result", "name": name, "result": result_str}
                tool_msg: dict[str, Any] = {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_str,
                }
                history.append(tool_msg)
                llm_messages.append(tool_msg)
            except Exception as exc:
                error_str = str(exc)
                yield {"type": "tool_error", "name": name, "error": error_str}
                error_msg: dict[str, Any] = {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": f"Error: {error_str}",
                }
                history.append(error_msg)
                llm_messages.append(error_msg)

        # Loop back for next LLM call with tool results injected

    else:
        logger.warning("Tool call loop hit max iterations (%d)", _MAX_TOOL_ITERATIONS)

    yield {"type": "done"}
