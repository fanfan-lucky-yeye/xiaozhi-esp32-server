import json
import logging
import openai
import requests
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LLM(ABC):
    @abstractmethod
    def response(self, conn, dialogue):
        """LLM response generator"""
        pass


class DeepSeekLLM(LLM):
    def __init__(self, config):
        self.model_name = config.get("model_name")
        self.api_key = config.get("api_key")
        self.base_url = config.get("url")
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

    def response(self, conn, dialogue):
        try:
            # 取最后一条用户消息
            last_msg = next(m for m in reversed(dialogue) if m["role"] == "user")

            # 获取 session_id（兼容 conn 为 None 的情况）
            session_id = getattr(conn, 'session_id', 'default_user') if conn else 'default_user'

            # 发起流式请求
            with requests.post(
                    f"{self.base_url}/chat-messages",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "query": last_msg["content"],
                        "response_mode": "streaming",
                        "user": session_id,
                        "inputs": {}
                    },
                    stream=True
            ) as r:
                for line in r.iter_lines():
                    if line.startswith(b'data: '):
                        event = json.loads(line[6:])
                        if event.get('answer'):
                            yield event['answer']

        except Exception as e:
            logger.error(f"DifyLLM response error: {e}")
            yield "【服务响应异常】"


class ChatGLMLLM(LLM):
    def __init__(self, config):
        self.model_name = config.get("model_name")
        self.api_key = config.get("api_key")
        self.base_url = config.get("url")
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

    def response(self, conn, dialogue):
        try:
            responses = self.client.chat.completions.create(
                model=self.model_name,
                messages=dialogue,
                stream=True
            )
            for chunk in responses:
                # 检查是否存在有效的choice且content不为空
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    content = getattr(delta, 'content', '')
                    if content:  # 仅在content非空时生成
                        yield content
        except Exception as e:
            logger.error(f"Error in response generation: {e}")


class DifyLLM(LLM):
    def __init__(self, config):
        self.api_key = config["api_key"]
        self.base_url = config.get("base_url", "https://api.dify.ai/v1").rstrip('/')

    def response(self, conn,dialogue):
        try:
            # 取最后一条用户消息
            last_msg = next(m for m in reversed(dialogue) if m["role"] == "user")

            # 发起流式请求
            with requests.post(
                    f"{self.base_url}/chat-messages",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "query": last_msg["content"],
                        "response_mode": "streaming",
                        "user": conn.session_id,
                        "inputs": {}
                    },
                    stream=True
            ) as r:
                for line in r.iter_lines():
                    if line.startswith(b'data: '):
                        event = json.loads(line[6:])
                        if event.get('answer'):
                            yield event['answer']

        except Exception:
            yield "【服务响应异常】"


def create_instance(class_name, *args, **kwargs):
    # 获取类对象
    cls_map = {
        "DeepSeekLLM": DeepSeekLLM,
        "ChatGLMLLM": ChatGLMLLM,
        "DifyLLM": DifyLLM,
        # 可扩展其他LLM实现
    }

    if cls := cls_map.get(class_name):
        return cls(*args, **kwargs)
    raise ValueError(f"不支持的LLM类型: {class_name}")
