"""RAG LLM."""
import asyncio
from typing import Any

from llama_index.core.constants import DEFAULT_CONTEXT_WINDOW
from llama_index.core.llms import (
    CompletionResponse,
    CompletionResponseGen,
    CustomLLM,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from pydantic import Field

from metagpt.config2 import config
from metagpt.provider.base_llm import BaseLLM
from metagpt.utils.async_helper import NestAsyncio
from metagpt.utils.token_counter import TOKEN_MAX


class RAGLLM(CustomLLM):
    """LlamaIndex's LLM is different from MetaGPT's LLM.

    Inherit CustomLLM from llamaindex, making MetaGPT's LLM can be used by LlamaIndex.

    Set context_length or max_token of LLM in config.yaml if you encounter "Calculated available context size -xxx was not non-negative" error.
    """

    model_infer: BaseLLM = Field(..., description="The MetaGPT's LLM.")
    context_window: int = -1
    num_output: int = -1
    model_name: str = ""

    def __init__(
        self,
        model_infer: BaseLLM,
        context_window: int = -1,
        num_output: int = -1,
        model_name: str = "",
        **data,
    ):
        """Custom initialization to compute sensible defaults then delegate field validation to Pydantic.

        The parent :class:`CustomLLM` (a Pydantic model) expects the declared fields to be provided when
        ``super().__init__`` is invoked. Previously these values were assigned *after* the parent init which
        caused a ``ValidationError`` because ``model_infer`` was missing. We now calculate the effective
        values first and pass them explicitly so that validation succeeds.
        """

        # Derive defaults from MetaGPT config if the caller leaves them as the sentinel ``-1``/empty string.
        if context_window < 0:
            context_window = TOKEN_MAX.get(config.llm.model, DEFAULT_CONTEXT_WINDOW)

        if num_output < 0:
            num_output = config.llm.max_token

        if not model_name:
            model_name = config.llm.model

        # Let Pydantic handle field assignment & validation.
        super().__init__(
            model_infer=model_infer,
            context_window=context_window,
            num_output=num_output,
            model_name=model_name,
            **data,
        )

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window, num_output=self.num_output, model_name=self.model_name or "unknown"
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        NestAsyncio.apply_once()
        return asyncio.get_event_loop().run_until_complete(self.acomplete(prompt, **kwargs))

    @llm_completion_callback()
    async def acomplete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        text = await self.model_infer.aask(msg=prompt, stream=False)
        return CompletionResponse(text=text)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        ...


def get_rag_llm(model_infer: BaseLLM = None) -> RAGLLM:
    """Get llm that can be used by LlamaIndex."""
    from metagpt.llm import LLM
    llm = LLM(config.llm)
    print("llm:", llm)

    return RAGLLM(model_infer=model_infer or llm)
