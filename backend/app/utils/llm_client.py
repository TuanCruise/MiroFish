"""
LLM client wrapper
Uniformly uses OpenAI format calls, supports reasoning models (reasoning_content field)
"""

import json
import re
from typing import Optional, Dict, Any, List
import httpx

from ..config import Config


class LLMClient:
    """LLM client - Called directly using httpx, compatible with reasoning models"""
    
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
            raise ValueError("LLM_API_KEY not configured")
        
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
        Send chat request
        
        Args:
            messages: List of messages
            temperature: Temperature parameter
            max_tokens: Maximum tokens
            response_format: Response format (e.g. JSON mode)
            
        Returns:
            Model response text
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
        
        # Some models (like MiniMax M2.5) include <think> reasoning in content, which needs to be removed
        content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
        return content
    
    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        Send chat request and return JSON
        
        Args:
            messages: List of messages
            temperature: Temperature parameter
            max_tokens: Maximum tokens
            
        Returns:
            Parsed JSON object
        """
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            # Don't send response_format - some models don't support it
            # response_format={"type": "json_object"}
        )
        # Clean up markdown code block tags
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
            raise ValueError("LLM returned empty response")

        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format returned by LLM: {cleaned_response[:500]}")
