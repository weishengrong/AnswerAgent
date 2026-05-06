import logging
from typing import Optional
from openai import AsyncOpenAI
from config.settings import llm_settings

logger = logging.getLogger(__name__)

_client: Optional[AsyncOpenAI] = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            base_url=llm_settings.LLM_BASE_URL,
            api_key=llm_settings.LLM_API_KEY
        )
    return _client


async def stream_llm(
    prompt: str,
    emitter,
    stream_id: str,
    event_type: str = "thinking_stream",
    system_prompt: Optional[str] = None
) -> str:
    client = _get_client()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    full_text = ""

    try:
        stream = await client.chat.completions.create(
            model=llm_settings.LLM_MODEL_NAME,
            messages=messages,
            stream=True
        )

        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                full_text += delta
                if emitter:
                    emitter.emit(event_type, {
                        "stream_id": stream_id,
                        "delta": delta,
                        "text": full_text
                    })
    except Exception as e:
        logger.error(f"❌ 流式 LLM 调用失败：{e}")
        if not full_text:
            raise

    return full_text
