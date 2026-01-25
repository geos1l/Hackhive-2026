"""LLM provider implementations for routing."""
from openai import OpenAI
from typing import Optional
import os


class LLMProvider:
    """Base class for LLM providers."""
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from LLM."""
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    """OpenAI provider (GPT-4, GPT-3.5, etc.)."""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4-turbo"):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key
            model_name: Model name (e.g., "gpt-4-turbo", "gpt-3.5-turbo")
        """
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate response using OpenAI.
        
        Args:
            prompt: User prompt
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            Generated response text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error with OpenAI: {str(e)}"


class AnthropicProvider(LLMProvider):
    """Anthropic provider (Claude models)."""
    
    def __init__(self, api_key: str, model_name: str = "claude-3-opus-20240229"):
        """
        Initialize Anthropic provider.
        
        Args:
            api_key: Anthropic API key
            model_name: Model name (e.g., "claude-3-opus-20240229", "claude-3-sonnet-20240229")
        """
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key)
            self.model_name = model_name
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate response using Anthropic.
        
        Args:
            prompt: User prompt
            **kwargs: Additional parameters (max_tokens, temperature, etc.)
            
        Returns:
            Generated response text
        """
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=kwargs.get("max_tokens", 1024),
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        except Exception as e:
            return f"Error with Anthropic: {str(e)}"


class OpenRouterProvider(LLMProvider):
    """
    OpenRouter provider (access to multiple models via OpenRouter).
    
    All OpenRouter models use OpenAI-compatible API, so this provider
    uses the OpenAI client with OpenRouter's base URL.
    Supports all OpenRouter models (Gemini, Llama, Kimi, GPT-OSS, Nemotron, etc.)
    """
    
    def __init__(self, api_key: str, model_name: str = "google/gemini-2.5-flash-lite"):
        """
        Initialize OpenRouter provider.
        
        Args:
            api_key: OpenRouter API key (can be different keys for different models)
            model_name: Model name (e.g., "google/gemini-2.5-flash-lite", 
                        "meta-llama/llama-3.1-405b-instruct", etc.)
        """
        # OpenRouter uses OpenAI-compatible API
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        self.model_name = model_name
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate response using OpenRouter.
        
        Args:
            prompt: User prompt
            **kwargs: Additional parameters
            
        Returns:
            Generated response text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error with OpenRouter: {str(e)}"


def create_provider(provider_name: str, model_name: str, api_key: str) -> LLMProvider:
    """
    Factory function to create appropriate LLM provider.
    
    Args:
        provider_name: "openai", "anthropic", or "openrouter"
        model_name: Model name
        api_key: API key
        
    Returns:
        LLMProvider instance
        
    Raises:
        ValueError: If provider_name is unknown
    """
    provider_name_lower = provider_name.lower()
    
    if provider_name_lower == "openai":
        return OpenAIProvider(api_key, model_name)
    elif provider_name_lower == "anthropic":
        return AnthropicProvider(api_key, model_name)
    elif provider_name_lower == "openrouter":
        return OpenRouterProvider(api_key, model_name)
    else:
        raise ValueError(f"Unknown provider: {provider_name}")
