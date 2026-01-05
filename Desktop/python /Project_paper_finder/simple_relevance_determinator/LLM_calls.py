# llm/deepseek_client.py
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Type

from pydantic import BaseModel, ValidationError
from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    model_name: str
    usage: Optional[dict] = None


class DeepSeekClient:
    """
    Minimal DeepSeek client.
    - OpenAI-compatible API
    - Stable JSON output for relevance judgement
    """

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        max_retries: int = 1,
        timeout_s: float = 60.0,
    ):
        self.model = model
        self.max_retries = max_retries

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_s,
        )

    async def call_pydantic(
        self,
        instructions: str,
        user_input: str,
        output_model: Type[BaseModel],
        temperature: float = 0.0,
    ) -> Tuple[BaseModel, LLMResponse]:
        

        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": user_input},
        ]

        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                )

                text = resp.choices[0].message.content

                json_str = self._extract_json(text)
                data = json.loads(json_str)

                parsed = output_model.model_validate(data)

                return parsed, LLMResponse(
                    model_name=self.model,
                    usage=resp.usage.model_dump() if resp.usage else None,
                )

            except (json.JSONDecodeError, ValidationError, KeyError) as e:
                # JSON / schema 问题：prompt 或模型输出问题
                logger.warning("DeepSeek JSON/schema error: %s", e)
                last_err = e
            except Exception as e:
                # 网络 / API 错误
                logger.warning("DeepSeek call failed: %s", e)
                last_err = e

            if attempt < self.max_retries:
                time.sleep(1.5 * (attempt + 1))

        raise RuntimeError(f"DeepSeek failed after retries: {last_err}")

    @staticmethod
    def _extract_json(text: str) -> str:
        """
        Extract first JSON object from model output.
        """
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in DeepSeek output")
        return text[start : end + 1]
