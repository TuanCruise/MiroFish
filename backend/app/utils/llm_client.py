"""
LLM客户端封装
统一使用OpenAI格式调用，支持reasoning模型（reasoning_content字段）
"""

import json
import re
from typing import Optional, Dict, Any, List
import httpx

from ..config import Config


class LLMClient:
    """LLM客户端 - 使用httpx直接调用，兼容reasoning模型"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model = model or Config.LLM_MODEL_NAME
        
        if not self.api_key:
            raise ValueError("LLM_API_KEY 未配置")
        
        # Use httpx with SSL verification disabled for self-signed certs
        self.http_client = httpx.Client(verify=False, timeout=300)
    
    def _get_url(self) -> str:
        """Build the chat completions URL"""
        base = self.base_url.rstrip('/')
        if base.endswith('/chat/completions'):
            return base
        return f"{base}/chat/completions"
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None
    ) -> str:
        """
        发送聊天请求
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            response_format: 响应格式（如JSON模式）
            
        Returns:
            模型响应文本
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if response_format:
            payload["response_format"] = response_format
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        url = self._get_url()
        response = self.http_client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        
        # Handle double-encoded JSON (API returns JSON string instead of object)
        if isinstance(data, str):
            import logging
            logging.getLogger('mirofish.llm').warning(f"API returned double-encoded JSON, re-parsing...")
            data = json.loads(data)
        
        message = data["choices"][0]["message"]
        
        # Try content first, then reasoning_content for reasoning models
        content = message.get("content")
        if not content:
            content = message.get("reasoning_content", "")
        if content is None:
            content = ""
        
        # 部分模型（如MiniMax M2.5）会在content中包含<think>思考内容，需要移除
        content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
        return content
    
    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        发送聊天请求并返回JSON
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            
        Returns:
            解析后的JSON对象
        """
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            # Don't send response_format - some models don't support it
            # response_format={"type": "json_object"}
        )
        # 清理markdown代码块标记
        cleaned_response = response.strip()
        cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response, flags=re.IGNORECASE)
        cleaned_response = re.sub(r'\n?```\s*$', '', cleaned_response)
        cleaned_response = cleaned_response.strip()
        
        # Try to extract JSON from the response if it contains other text
        if cleaned_response and not cleaned_response.startswith('{'):
            # Look for JSON object in the response
            json_match = re.search(r'\{[\s\S]*\}', cleaned_response)
            if json_match:
                cleaned_response = json_match.group(0)

        if not cleaned_response:
            raise ValueError("LLM返回空响应")

        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            raise ValueError(f"LLM返回的JSON格式无效: {cleaned_response[:500]}")
