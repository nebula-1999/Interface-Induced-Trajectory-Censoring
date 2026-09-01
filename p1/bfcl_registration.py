#!/usr/bin/env python3
"""Register P1's local OpenAI-compatible endpoint with BFCL.

BFCL's stock OSS Qwen handler calls ``/v1/completions`` and parses raw text in
the benchmark process.  That path deliberately bypasses the serving parser P1
is meant to measure.  This adapter instead reuses BFCL's official OpenAI FC
handler, adding only an explicit ``tool_choice=auto`` and join keys in HTTP
headers.  Dataset loading, tool execution, and scoring remain BFCL code.
"""
from __future__ import annotations

import os

from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.constants.model_config import ModelConfig, MODEL_CONFIG_MAPPING
from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.model_handler.api_inference.openai_completion import (
    OpenAICompletionsHandler,
)
from bfcl_eval.model_handler.utils import convert_to_tool


class P1OpenAIHandler(OpenAICompletionsHandler):
    """BFCL OpenAI FC handler with explicit protocol and trace join keys."""

    def _pre_query_processing_FC(self, inference_data, test_entry):
        inference_data = super()._pre_query_processing_FC(inference_data, test_entry)
        inference_data["_p1_case_id"] = test_entry["id"]
        inference_data["_p1_request_index"] = 0
        return inference_data

    def _compile_tools(self, inference_data, test_entry):
        # Keep this explicit so a BFCL upgrade cannot silently switch schemas.
        inference_data["tools"] = convert_to_tool(
            test_entry["function"], GORILLA_TO_OPENAPI, ModelStyle.OPENAI_COMPLETIONS
        )
        return inference_data

    def _query_FC(self, inference_data):
        messages = inference_data["message"]
        tools = inference_data["tools"]
        request_index = inference_data["_p1_request_index"]
        inference_data["_p1_request_index"] = request_index + 1
        inference_data["inference_input_log"] = {
            "case_id": inference_data["_p1_case_id"],
            "request_index": request_index,
            "message": repr(messages),
            "tools": tools,
            "tool_choice": "auto",
        }
        kwargs = {
            "messages": messages,
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": int(os.environ.get("P1_MAX_TOKENS", "2048")),
            "seed": int(os.environ.get("P1_SEED", "0")),
            "store": False,
            "extra_headers": {
                "X-P1-Case-ID": inference_data["_p1_case_id"],
                "X-P1-Request-Index": str(request_index),
            },
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        return self.generate_with_backoff(**kwargs)


def register() -> str:
    registry_name = os.environ.get(
        "P1_BFCL_MODEL_ID", "P1-Qwen2.5-Coder-7B-Hermes-FC"
    )
    served_name = os.environ.get("P1_SERVED_MODEL", "p1-qwen25-coder-7b")
    MODEL_CONFIG_MAPPING[registry_name] = ModelConfig(
        model_name=served_name,
        display_name=registry_name,
        url="https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct",
        org="Qwen",
        license="apache-2.0",
        model_handler=P1OpenAIHandler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=True,
    )
    return registry_name


REGISTERED_MODEL = register()
