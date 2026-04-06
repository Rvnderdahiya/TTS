from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import find_dotenv, load_dotenv


SECTION_SYNTHESIS_PROMPT = """
You are an expert analyst who prepares grounded notes for a two-host podcast.
Use only the provided context and never invent facts.

Document profile:
{document_profile}

Coverage plan:
{coverage_plan}

Conversation history:
{history}

Current context chunk:
{chunk}

User query:
{user_query}

Answer:
"""

FINAL_DIALOGUE_PROMPT = """
You are writing a realistic two-host podcast that teaches a listener the document thoroughly.
The conversation must sound human and natural, but remain fully grounded.

Host names:
- Primary host: {host_primary}
- Secondary host: {host_secondary}

Document profile:
{document_profile}

Coverage plan:
{coverage_plan}

Synthesized notes:
{notes}

User query:
{user_query}

Write only dialogue.
Format every line as:
Speaker: text [section_id pp.start-end]
"""


class SafeChainPodcastGenerator:
    def __init__(self, model_selector: str, strict_mode: bool) -> None:
        self.model_selector = model_selector
        self.strict_mode = strict_mode

    def generate(self, document: dict[str, Any], plan: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
        try:
            return self._generate_with_safechain(document, plan, options)
        except Exception as exc:
            if self.strict_mode:
                raise
            return self._fallback(document, plan, options, str(exc))

    def _generate_with_safechain(self, document: dict[str, Any], plan: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
        # Keep the bootstrap aligned with the SafeChain pattern you showed.
        env_path = find_dotenv("example.env")
        if env_path:
            load_dotenv(env_path)
        os.environ["CONFIG_PATH"] = os.getenv("CONFIG_PATH", "./config/config.yml")

        import nest_asyncio

        nest_asyncio.apply()

        from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate  # noqa: F401
        from safechain.utils import get_token_from_env  # noqa: F401
        from safechain.lcel import model
        from safechain.prompts import ValidPromptTemplate
        from langchain.schema import StrOutputParser

        llm = model(self.model_selector)
        document_profile = json.dumps(
            {
                "title": document["title"],
                "page_count": document["page_count"],
                "word_count": document["word_count"],
            },
            indent=2,
        )
        coverage_plan = json.dumps(plan["coverage_items"], indent=2)
        user_query = options.get("listener_goal", "Turn this document into a human-style podcast dialogue.")

        prompt = ValidPromptTemplate(
            input_variables=["chunk", "history", "user_query", "document_profile", "coverage_plan"],
            template=SECTION_SYNTHESIS_PROMPT,
        )
        chain = prompt | llm | StrOutputParser()

        all_summaries: list[str] = []
        prev_call_summary = ""
        for chunk in _chunk_text(json.dumps(document["sections"], indent=2)):
            merged_chunk = f"{prev_call_summary}\n\n{chunk}" if prev_call_summary else chunk
            response = chain.invoke(
                {
                    "chunk": merged_chunk,
                    "history": prev_call_summary,
                    "user_query": user_query,
                    "document_profile": document_profile,
                    "coverage_plan": coverage_plan,
                }
            )
            all_summaries.append(response)
            prev_call_summary = "\n".join(response.strip().splitlines()[-8:])

        dialogue_prompt = ValidPromptTemplate(
            input_variables=["notes", "user_query", "document_profile", "coverage_plan", "host_primary", "host_secondary"],
            template=FINAL_DIALOGUE_PROMPT,
        )
        dialogue_chain = dialogue_prompt | llm | StrOutputParser()
        dialogue_text = dialogue_chain.invoke(
            {
                "notes": "\n".join(all_summaries),
                "user_query": user_query,
                "document_profile": document_profile,
                "coverage_plan": coverage_plan,
                "host_primary": options["host_primary"],
                "host_secondary": options["host_secondary"],
            }
        )

        return {
            "title": plan["episode_title"],
            "episode_summary": "Dialogue drafted with SafeChain.",
            "dialogue": _parse_dialogue(dialogue_text, options["host_primary"], options["host_secondary"]),
            "warnings": [],
            "engine": "safechain",
            "audio_status": "not_generated",
        }

    def _fallback(self, document: dict[str, Any], plan: dict[str, Any], options: dict[str, Any], reason: str) -> dict[str, Any]:
        host_primary = options["host_primary"]
        host_secondary = options["host_secondary"]
        dialogue: list[dict[str, Any]] = [
            {
                "speaker": host_primary,
                "text": f"Today we are unpacking {document['title']}. We will cover the whole document in a way that is easy to retain.",
                "citations": [],
            },
            {
                "speaker": host_secondary,
                "text": "So this is not a quick summary. We are walking section by section and clarifying what each part is doing.",
                "citations": [],
            },
        ]
        for item in plan["coverage_items"]:
            citation = f"{item['section_id']} pp.{item['page_start']}-{item['page_end']}"
            dialogue.append(
                {
                    "speaker": host_primary,
                    "text": f"Let us start with {item['title']}. {item['summary']}",
                    "citations": [citation],
                }
            )
            dialogue.append(
                {
                    "speaker": host_secondary,
                    "text": "The important question here is what the listener should remember after hearing this part once.",
                    "citations": [citation],
                }
            )

        return {
            "title": plan["episode_title"],
            "episode_summary": "Fallback draft generated so the local flow can be tested before the real SafeChain package is connected.",
            "dialogue": dialogue,
            "warnings": [f"SafeChain unavailable locally. Fallback draft used: {reason}"],
            "engine": "fallback-draft",
            "audio_status": "not_generated",
        }


def _chunk_text(text: str, max_words: int = 1200, overlap: int = 200) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    step = max(max_words - overlap, 1)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + max_words]).strip()
        if chunk:
            chunks.append(chunk)
        if start + max_words >= len(words):
            break
    return chunks


def _parse_dialogue(raw_text: str, host_primary: str, host_secondary: str) -> list[dict[str, Any]]:
    allowed_speakers = {host_primary, host_secondary, "Host", "Narrator", "Analyst", "Guide"}
    parsed: list[dict[str, Any]] = []
    for line in raw_text.splitlines():
        match = re.match(r"^(?P<speaker>[^:]{2,40}):\s*(?P<body>.+)$", line.strip())
        if not match:
            continue
        speaker = match.group("speaker").strip()
        body = match.group("body").strip()
        citations = re.findall(r"\[(.*?)\]", body)
        body = re.sub(r"\s*\[(.*?)\]\s*", " ", body).strip()
        if speaker not in allowed_speakers:
            speaker = host_primary if len(parsed) % 2 == 0 else host_secondary
        parsed.append({"speaker": speaker, "text": body, "citations": citations})
        return parsed


def diagnose_safechain(model_selector: str) -> dict[str, Any]:
    env_file_name = "example.env"
    env_path = find_dotenv(env_file_name, usecwd=True)
    diagnosis: dict[str, Any] = {
        "safechain_env_file": env_file_name,
        "safechain_env_resolved": env_path or None,
        "config_path": os.getenv("CONFIG_PATH", "./config/config.yml"),
        "model_selector": model_selector,
        "safechain_importable": False,
        "valid_prompt_template_importable": False,
        "str_output_parser_importable": False,
        "model_callable_importable": False,
        "errors": [],
    }

    if env_path:
        load_dotenv(env_path)

    try:
        from safechain.lcel import model  # noqa: F401

        diagnosis["safechain_importable"] = True
        diagnosis["model_callable_importable"] = True
    except Exception as exc:
        diagnosis["errors"].append(f"safechain.lcel.model import failed: {exc}")

    try:
        from safechain.prompts import ValidPromptTemplate  # noqa: F401

        diagnosis["valid_prompt_template_importable"] = True
    except Exception as exc:
        diagnosis["errors"].append(f"safechain.prompts.ValidPromptTemplate import failed: {exc}")

    try:
        from langchain.schema import StrOutputParser  # noqa: F401

        diagnosis["str_output_parser_importable"] = True
    except Exception:
        try:
            from langchain_core.output_parsers import StrOutputParser  # noqa: F401

            diagnosis["str_output_parser_importable"] = True
        except Exception as exc:
            diagnosis["errors"].append(f"StrOutputParser import failed: {exc}")

    return diagnosis
