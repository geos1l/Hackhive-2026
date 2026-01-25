"""Router AI service using Google Gemini Flash 2.5 Lite via OpenRouter."""
from openai import OpenAI
from typing import Optional


class RouterAI:
    """
    Router AI using Gemini Flash 2.5 Lite via OpenRouter.
    
    For now, acts as a simple chatbot. Later will route to other LLMs via MCP.
    """
    
    def __init__(self, api_key: str, model_name: str = "google/gemini-2.5-flash-lite"):
        """
        Initialize Router AI with Gemini via OpenRouter.
        
        Args:
            api_key: OpenRouter API key
            model_name: Model to use (default: google/gemini-2.5-flash-lite)
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        self.model_name = model_name
        
    def process(self, user_text: str) -> str:
        """
        Process user text and return AI response.
        
        Args:
            user_text: Text input from user (from STT)
            
        Returns:
            Response text from Gemini
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": user_text}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            # Return error message if API call fails
            return f"Error processing request: {str(e)}"
    
    def close(self):
        """Close the client connection."""
        # OpenAI client doesn't need explicit close, but keeping for consistency
        pass