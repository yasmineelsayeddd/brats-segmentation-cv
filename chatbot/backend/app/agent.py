from __future__ import annotations

import json
import re
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from app.config import agent_config, llm_config
from app.tools import (
    analyze_uncertainty,
    cascade_detect,
    compute_metrics,
    explain_findings,
    segment_scan,
)

_segment_scan = tool(segment_scan)
_analyze_uncertainty = tool(analyze_uncertainty)
_compute_metrics = tool(compute_metrics)
_cascade_detect = tool(cascade_detect)
_explain_findings = tool(explain_findings)

TOOLS = [_segment_scan, _analyze_uncertainty, _compute_metrics, _cascade_detect, _explain_findings]


class AgentState(TypedDict):
    messages: list
    session_id: str | None


def _build_system_message() -> str:
    return agent_config().get("system_prompt", "You are a medical imaging assistant.")


class BraTSAgent:
    def __init__(self):
        self._llm = None
        self._graph = None

    @property
    def llm(self):
        if self._llm is None:
            cfg = llm_config()
            api_key = cfg.get("api_key", "")
            if not api_key:
                raise ValueError(
                    "OpenRouter API key not configured. Set 'llm.api_key' in chatbot/config.yaml"
                )
            self._llm = ChatOpenAI(
                model=cfg.get("model"),
                temperature=cfg.get("temperature", 0.3),
                max_tokens=cfg.get("max_tokens", 2048),
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
                default_headers={"HTTP-Referer": "https://github.com", "X-Title": "BraTS Chatbot"},
            ).bind_tools(TOOLS)
        return self._llm

    @property
    def graph(self):
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph

    def _route(self, state: AgentState) -> Literal["tools", "end"]:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "end"

    def _call_model(self, state: AgentState) -> dict:
        system_msg = SystemMessage(content=_build_system_message())
        messages = [system_msg] + state["messages"]
        response = self.llm.invoke(messages)
        return {"messages": [response]}

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        tool_node = ToolNode(TOOLS)
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", tool_node)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", self._route, {"tools": "tools", "end": END})
        workflow.add_edge("tools", "agent")
        return workflow.compile()

    def _generate_suggestions(self, user_msg: str, assistant_msg: str) -> list[str]:
        cfg = llm_config()
        api_key = cfg.get("api_key", "")
        if not api_key:
            return []
        try:
            llm = ChatOpenAI(
                model=cfg.get("model"),
                temperature=0.5,
                max_tokens=300,
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
            prompt = (
                "Based on this brain tumor analysis conversation, suggest 3 concise follow-up questions "
                "the user might want to ask next. Return ONLY a JSON array of 3 strings, no other text.\n\n"
                f"User: {user_msg}\n"
                f"Assistant: {assistant_msg}"
            )
            response = llm.invoke([
                SystemMessage(content="You suggest relevant follow-up questions for a brain tumor MRI analysis assistant."),
                HumanMessage(content=prompt),
            ])
            text = response.content.strip()
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            suggestions = json.loads(text)
            if isinstance(suggestions, list) and len(suggestions) > 0:
                return [str(s) for s in suggestions[:3]]
        except Exception:
            pass
        return []

    def invoke(self, messages: list[dict], session_id: str | None = None):
        langchain_msgs = []
        for msg in messages:
            if msg["role"] == "user":
                langchain_msgs.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_msgs.append(AIMessage(content=msg["content"]))

        state = {"messages": langchain_msgs, "session_id": session_id}
        result = self.graph.invoke(state)

        assistant_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
        tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]

        tool_calls_info = []
        for tm in tool_messages:
            tool_calls_info.append({
                "tool_name": tm.name,
                "result_summary": tm.content[:200] if tm.content else "",
            })

        final_content = ""
        if assistant_messages:
            final_content = assistant_messages[-1].content

        suggestions = []
        user_msgs = [m for m in messages if m["role"] == "user"]
        if user_msgs and final_content:
            suggestions = self._generate_suggestions(user_msgs[-1]["content"], final_content)

        return {
            "content": final_content,
            "tool_calls": tool_calls_info,
            "suggestions": suggestions,
        }

    def stream(self, messages: list[dict], session_id: str | None = None):
        langchain_msgs = []
        for msg in messages:
            if msg["role"] == "user":
                langchain_msgs.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_msgs.append(AIMessage(content=msg["content"]))

        state = {"messages": langchain_msgs, "session_id": session_id}

        for event in self.graph.stream(state, stream_mode="values"):
            last = event["messages"][-1]
            if isinstance(last, AIMessage):
                if last.tool_calls:
                    for tc in last.tool_calls:
                        yield {
                            "type": "tool_call",
                            "tool_call": {
                                "tool_name": tc["name"],
                                "tool_input": tc["args"],
                                "result_summary": "Running...",
                            },
                        }
                if last.content:
                    yield {"type": "content", "content": last.content}

        yield {"type": "done", "done": True}


agent = BraTSAgent()
