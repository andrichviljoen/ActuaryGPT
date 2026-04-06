from __future__ import annotations

import json
from dataclasses import dataclass

from openai import OpenAI

from reserving_app.core.config import CONFIG


@dataclass
class AIContext:
    mapping: dict
    assumptions: dict
    reserve_summary: dict
    diagnostics_summary: dict
    chart_summary: dict

    def to_prompt_payload(self) -> str:
        return json.dumps(
            {
                "mapping": self.mapping,
                "assumptions": self.assumptions,
                "reserve_summary": self.reserve_summary,
                "diagnostics_summary": self.diagnostics_summary,
                "chart_summary": self.chart_summary,
            },
            indent=2,
            default=str,
        )


def resolve_ai_request_state(current_question: str, preset_question: str | None, ask_clicked: bool) -> tuple[str, bool]:
    question = current_question or ""
    should_submit = False
    if preset_question:
        question = preset_question
        should_submit = True
    elif ask_clicked and question.strip():
        should_submit = True
    return question, should_submit


def ask_assistant(question: str, context: AIContext) -> str:
    if not CONFIG.openai_api_key:
        return "OpenAI API key not configured. Add OPENAI_API_KEY in your environment to use AI Assistant."

    client = OpenAI(api_key=CONFIG.openai_api_key)
    system_prompt = (
        "You are an actuarial reserving assistant. "
        "Use only values provided in context. If missing, state limitation explicitly. "
        "Never fabricate numeric outputs."
    )

    completion = client.chat.completions.create(
        model=CONFIG.openai_model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Context JSON:\n{context.to_prompt_payload()}\n\nQuestion:\n{question}",
            },
        ],
    )
    return completion.choices[0].message.content or "No response generated."
