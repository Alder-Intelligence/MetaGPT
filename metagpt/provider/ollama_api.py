#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : self-host open llm model with ollama which isn't openai-api-compatible

import json
import re
from enum import Enum, auto
from typing import AsyncGenerator, Optional, Tuple

from metagpt.configs.llm_config import LLMConfig, LLMType
from metagpt.const import USE_CONFIG_TIMEOUT
from metagpt.logs import log_llm_stream, log_llm_stream_thinking
from metagpt.provider.base_llm import BaseLLM
from metagpt.provider.general_api_requestor import GeneralAPIRequestor, OpenAIResponse
from metagpt.provider.llm_provider_registry import register_provider
from metagpt.utils.cost_manager import TokenCostManager


class OllamaMessageAPI(Enum):
    # default
    CHAT = auto()
    GENERATE = auto()
    EMBED = auto()
    EMBEDDINGS = auto()


class OllamaMessageBase:
    api_type = OllamaMessageAPI.CHAT

    def __init__(self, model: str, **additional_kwargs) -> None:
        self.model, self.additional_kwargs = model, additional_kwargs
        self._image_b64_rms = len("data:image/jpeg;base64,")

    @property
    def api_suffix(self) -> str:
        raise NotImplementedError

    def apply(self, messages: list[dict]) -> dict:
        raise NotImplementedError

    def decode(self, response: OpenAIResponse) -> dict:
        text = response.data.decode("utf-8")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Fallback for servers that return NDJSON concatenated content when stream=True
            # Parse line by line and return the last complete JSON object
            obj = None
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
            if obj is not None:
                return obj
            raise

    def get_choice(self, to_choice_dict: dict) -> str:
        raise NotImplementedError

    def _parse_input_msg(self, msg: dict) -> Tuple[Optional[str], Optional[str]]:
        if "type" in msg:
            tpe = msg["type"]
            if tpe == "text":
                return msg["text"], None
            elif tpe == "image_url":
                return None, msg["image_url"]["url"][self._image_b64_rms :]
            else:
                raise ValueError
        else:
            raise ValueError


class OllamaMessageMeta(type):
    registed_message = {}

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        for base in bases:
            if issubclass(base, OllamaMessageBase):
                api_type = attrs["api_type"]
                assert api_type not in OllamaMessageMeta.registed_message, "api_type already exist"
                assert isinstance(api_type, OllamaMessageAPI), "api_type not support"
                OllamaMessageMeta.registed_message[api_type] = cls

    @classmethod
    def get_message(cls, input_type: OllamaMessageAPI) -> type[OllamaMessageBase]:
        return cls.registed_message[input_type]


class OllamaMessageChat(OllamaMessageBase, metaclass=OllamaMessageMeta):
    api_type = OllamaMessageAPI.CHAT

    @property
    def api_suffix(self) -> str:
        return "/chat"

    def apply(self, messages: list[dict]) -> dict:
        content = messages[0]["content"]
        prompts = []
        images = []
        if isinstance(content, list):
            for msg in content:
                prompt, image = self._parse_input_msg(msg)
                if prompt:
                    prompts.append(prompt)
                if image:
                    images.append(image)
        else:
            prompts.append(content)
        messes = []
        for prompt in prompts:
            if len(images) > 0:
                messes.append({"role": "user", "content": prompt, "images": images})
            else:
                messes.append({"role": "user", "content": prompt})
        sends = {"model": self.model, "messages": messes}
        sends.update(self.additional_kwargs)
        return sends

    def get_choice(self, to_choice_dict: dict) -> str:
        message = to_choice_dict["message"]
        if message["role"] == "assistant":
            return message["content"]
        else:
            raise ValueError


class OllamaMessageGenerate(OllamaMessageChat, metaclass=OllamaMessageMeta):
    api_type = OllamaMessageAPI.GENERATE

    @property
    def api_suffix(self) -> str:
        return "/generate"

    def apply(self, messages: list[dict]) -> dict:
        content = messages[0]["content"]
        prompts = []
        images = []
        if isinstance(content, list):
            for msg in content:
                prompt, image = self._parse_input_msg(msg)
                if prompt:
                    prompts.append(prompt)
                if image:
                    images.append(image)
        else:
            prompts.append(content)
        if len(images) > 0:
            sends = {"model": self.model, "prompt": "\n".join(prompts), "images": images}
        else:
            sends = {"model": self.model, "prompt": "\n".join(prompts)}
        sends.update(self.additional_kwargs)
        return sends

    def get_choice(self, to_choice_dict: dict) -> str:
        return to_choice_dict["response"]


class OllamaMessageEmbeddings(OllamaMessageBase, metaclass=OllamaMessageMeta):
    api_type = OllamaMessageAPI.EMBEDDINGS

    @property
    def api_suffix(self) -> str:
        return "/embeddings"

    def apply(self, messages: list[dict]) -> dict:
        content = messages[0]["content"]
        prompts = []  # NOTE: not support image to embedding
        if isinstance(content, list):
            for msg in content:
                prompt, _ = self._parse_input_msg(msg)
                if prompt:
                    prompts.append(prompt)
        else:
            prompts.append(content)
        sends = {"model": self.model, "prompt": "\n".join(prompts)}
        sends.update(self.additional_kwargs)
        return sends


class OllamaMessageEmbed(OllamaMessageEmbeddings, metaclass=OllamaMessageMeta):
    api_type = OllamaMessageAPI.EMBED

    @property
    def api_suffix(self) -> str:
        return "/embed"

    def apply(self, messages: list[dict]) -> dict:
        content = messages[0]["content"]
        prompts = []  # NOTE: not support image to embedding
        if isinstance(content, list):
            for msg in content:
                prompt, _ = self._parse_input_msg(msg)
                if prompt:
                    prompts.append(prompt)
        else:
            prompts.append(content)
        sends = {"model": self.model, "input": prompts}
        sends.update(self.additional_kwargs)
        return sends


@register_provider(LLMType.OLLAMA)
class OllamaLLM(BaseLLM):
    """
    Refs to `https://github.com/jmorganca/ollama/blob/main/docs/api.md#generate-a-chat-completion`
    """

    def __init__(self, config: LLMConfig):
        self.client = GeneralAPIRequestor(base_url=config.base_url, key=config.api_key)
        self.config = config
        self.http_method = "post"
        self.use_system_prompt = False
        self.cost_manager = TokenCostManager()
        self.__init_ollama(config)

    @property
    def _llama_api_inuse(self) -> OllamaMessageAPI:
        return OllamaMessageAPI.CHAT

    @property
    def _llama_api_kwargs(self) -> dict:
        return {"options": {"temperature": 0.3}, "stream": self.config.stream}

    def __init_ollama(self, config: LLMConfig):
        assert config.base_url, "ollama base url is required!"
        self.model = config.model
        self.pricing_plan = self.model
        ollama_message = OllamaMessageMeta.get_message(self._llama_api_inuse)
        self.ollama_message = ollama_message(model=self.model, **self._llama_api_kwargs)

    def get_usage(self, resp: dict) -> dict:
        return {"prompt_tokens": resp.get("prompt_eval_count", 0), "completion_tokens": resp.get("eval_count", 0)}

    async def _achat_completion(self, messages: list[dict], timeout: int = USE_CONFIG_TIMEOUT) -> dict:
        payload = self.ollama_message.apply(messages=messages)
        if isinstance(payload, dict):
            # Force non-streaming payload to ensure a single JSON object response
            payload["stream"] = False
        resp, _, _ = await self.client.arequest(
            method=self.http_method,
            url=self.ollama_message.api_suffix,
            params=payload,
            request_timeout=self.get_timeout(timeout),
        )
        if isinstance(resp, AsyncGenerator):
            return await self._processing_openai_response_async_generator(resp)
        elif isinstance(resp, OpenAIResponse):
            return self._processing_openai_response(resp)
        else:
            raise ValueError

    def get_choice_text(self, rsp):
        # Prefer provider-specific reasoning fields if present
        if isinstance(rsp, dict):
            message = rsp.get("message")
            if isinstance(message, dict):
                thinking = message.get("thinking")
                if thinking:
                    self.reasoning_content = thinking

        text = self.ollama_message.get_choice(rsp)
        cleaned, reasoning = self._extract_reasoning(text)
        if reasoning:
            self.reasoning_content = reasoning
        return cleaned

    async def acompletion(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT) -> dict:
        return await self._achat_completion(messages, timeout=self.get_timeout(timeout))

    async def _achat_completion_stream(self, messages: list[dict], timeout: int = USE_CONFIG_TIMEOUT) -> str:
        payload = self.ollama_message.apply(messages=messages)
        # Force streaming response at HTTP and payload levels
        if isinstance(payload, dict):
            payload["stream"] = True
        resp, _, _ = await self.client.arequest(
            method=self.http_method,
            url=self.ollama_message.api_suffix,
            params=payload,
            request_timeout=self.get_timeout(timeout),
            stream=True,
        )
        if isinstance(resp, AsyncGenerator):
            return await self._processing_openai_response_async_generator(resp)
        elif isinstance(resp, OpenAIResponse):
            return self._processing_openai_response(resp)
        else:
            raise ValueError

    def _processing_openai_response(self, openai_resp: OpenAIResponse):
        resp = self.ollama_message.decode(openai_resp)
        usage = self.get_usage(resp)
        self._update_costs(usage)
        return resp

    async def _processing_openai_response_async_generator(self, ag_openai_resp: AsyncGenerator[OpenAIResponse, None]):
        collected_content = []
        usage = {}
        reasoning_chunks: list[str] = []
        async for raw_chunk in ag_openai_resp:
            chunk = self.ollama_message.decode(raw_chunk)

            if not chunk.get("done", False):
                content = self.ollama_message.get_choice(chunk)
                collected_content.append(content)
                log_llm_stream(content)
            else:
                # stream finished
                usage = self.get_usage(chunk)
            # capture provider-specific reasoning if present in any chunk
            if isinstance(chunk, dict):
                message = chunk.get("message")
                if isinstance(message, dict):
                    thinking = message.get("thinking")
                    if thinking:
                        reasoning_chunks.append(thinking)
                        log_llm_stream_thinking(thinking)
        log_llm_stream("\n")

        self._update_costs(usage)
        full_content = "".join(collected_content)
        cleaned, reasoning = self._extract_reasoning(full_content)
        if reasoning:
            self.reasoning_content = reasoning
        elif reasoning_chunks:
            self.reasoning_content = "".join(reasoning_chunks).strip()
        return cleaned

    @staticmethod
    def _extract_reasoning(text: str) -> tuple[str, Optional[str]]:
        """Extract reasoning/thinking content from model output and return (cleaned, reasoning).

        Heuristics:
        - Prefer XML-like tags often used by reasoning models: <think>...</think>, <reasoning>...</reasoning>, <analysis>...</analysis>
        - If multiple matches exist, concatenate them in order of appearance.
        - Remove matched segments from the final answer.
        """
        if not text:
            return text, None

        patterns = [
            re.compile(r"<think>([\s\S]*?)</think>", re.IGNORECASE),
            re.compile(r"<reasoning>([\s\S]*?)</reasoning>", re.IGNORECASE),
            re.compile(r"<analysis>([\s\S]*?)</analysis>", re.IGNORECASE),
        ]

        reasoning_parts: list[str] = []
        cleaned = text
        for pat in patterns:
            matches = list(pat.finditer(cleaned))
            if matches:
                for m in matches:
                    reasoning_parts.append(m.group(1).strip())
                cleaned = pat.sub("", cleaned)

        reasoning_text = "\n\n".join([p for p in reasoning_parts if p]) if reasoning_parts else None
        return cleaned.strip(), reasoning_text


@register_provider(LLMType.OLLAMA_GENERATE)
class OllamaGenerate(OllamaLLM):
    @property
    def _llama_api_inuse(self) -> OllamaMessageAPI:
        return OllamaMessageAPI.GENERATE

    @property
    def _llama_api_kwargs(self) -> dict:
        return {"options": {"temperature": 0.3}, "stream": self.config.stream}


@register_provider(LLMType.OLLAMA_EMBEDDINGS)
class OllamaEmbeddings(OllamaLLM):
    @property
    def _llama_api_inuse(self) -> OllamaMessageAPI:
        return OllamaMessageAPI.EMBEDDINGS

    @property
    def _llama_api_kwargs(self) -> dict:
        return {"options": {"temperature": 0.3}}

    @property
    def _llama_embedding_key(self) -> str:
        return "embedding"

    async def _achat_completion(self, messages: list[dict], timeout: int = USE_CONFIG_TIMEOUT) -> dict:
        payload = self.ollama_message.apply(messages=messages)
        # Force non-streaming response at HTTP and payload levels
        if isinstance(payload, dict):
            payload["stream"] = False
        resp, _, _ = await self.client.arequest(
            method=self.http_method,
            url=self.ollama_message.api_suffix,
            params=payload,
            request_timeout=self.get_timeout(timeout),
        )
        return self.ollama_message.decode(resp)[self._llama_embedding_key]

    async def _achat_completion_stream(self, messages: list[dict], timeout: int = USE_CONFIG_TIMEOUT) -> str:
        return await self._achat_completion(messages, timeout=self.get_timeout(timeout))

    def get_choice_text(self, rsp):
        return rsp


@register_provider(LLMType.OLLAMA_EMBED)
class OllamaEmbed(OllamaEmbeddings):
    @property
    def _llama_api_inuse(self) -> OllamaMessageAPI:
        return OllamaMessageAPI.EMBED

    @property
    def _llama_embedding_key(self) -> str:
        return "embeddings"
