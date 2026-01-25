"""Router AI with Supabase integration for intelligent LLM routing."""
from openai import OpenAI
from typing import Optional, Dict
import re

from .supabase_client import SupabaseBenchmarkDB
from .llm_providers import create_provider, LLMProvider
from config.settings import Settings


class RouterAI:
    """
    Router AI that:
    1. Uses Gemini to classify user prompt into one of 6 categories
    2. Queries Supabase to get best model for that category
    3. Routes prompt to appropriate LLM using API keys from .env
    4. Returns response (which gets converted to speech)
    """
    
    def __init__(
        self,
        router_api_key: str,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        router_model: str = "google/gemini-2.5-flash-lite"
    ):
        """
        Initialize Router AI.
        
        Args:
            router_api_key: OpenRouter API key (for Gemini classification)
            supabase_url: Supabase project URL (defaults to Settings)
            supabase_key: Supabase API key (defaults to Settings)
            router_model: Model for classification (default: Gemini Flash 2.5 Lite)
        """
        # Router uses Gemini via OpenRouter for classification
        self.router_client = OpenAI(
            api_key=router_api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        self.router_model = router_model
        
        # Supabase connection
        supabase_url = supabase_url or Settings.SUPABASE_URL
        supabase_key = supabase_key or Settings.SUPABASE_KEY
        self.benchmark_db = SupabaseBenchmarkDB(supabase_url, supabase_key)
        
        # Cache for provider instances (avoid recreating)
        self._provider_cache = {}
        
        # Cache for categories (avoid repeated queries)
        self._categories_cache = None
        
        # Map Supabase best_model labels to OpenRouter model IDs + API key env vars.
        # Add aliases here to match whatever is stored in Supabase.
        self._model_routing_map = {
            "kimi k2 thinking": {
                "model": "moonshotai/kimi-k2",
                "api_key_env": "KIMI_API_KEY",
            },
            "gpt oss 120 b": {
                "model": "openai/gpt-oss-120b",
                "api_key_env": "GPT_OSS_API_KEY",
            },
            "nematron ultra 253b": {
                "model": "nvidia/nemotron-4-340b-instruct",
                "api_key_env": "NEMOTRON_API_KEY",
            },
            "llama 3.51 405 b": {
                "model": "meta-llama/llama-3.1-405b-instruct",
                "api_key_env": "LLAMA_API_KEY",
            },
            "gemini flash 2.5 lite": {
                "model": "google/gemini-2.5-flash-lite",
                "api_key_env": "GEMINI_API_KEY",
            },
        }
    
    def _get_categories(self) -> list[str]:
        """Get categories from Supabase (with caching)."""
        if self._categories_cache is None:
            self._categories_cache = self.benchmark_db.get_all_categories()
        return self._categories_cache
    
    def _classify_category(self, user_text: str) -> Optional[str]:
        """
        Use Gemini to classify user prompt into a category.
        
        Args:
            user_text: User prompt/text
            
        Returns:
            Category name or None if classification fails
        """
        # Get available categories from Supabase
        categories = self._get_categories()
        if not categories:
            print("Warning: No categories found in database")
            return None
        
        categories_str = ", ".join(categories)
        
        classification_prompt = f"""You are a prompt classifier. Analyze the following user prompt and determine which category it best fits into.

Available categories: {categories_str}

User prompt: "{user_text}"

Respond with ONLY the category name (exactly as listed above). Do not include any explanation, quotes, or additional text. Just the category name."""

        try:
            response = self.router_client.chat.completions.create(
                model=self.router_model,
                messages=[{"role": "user", "content": classification_prompt}],
                temperature=0.1  # Low temperature for consistent classification
            )
            category = response.choices[0].message.content.strip()
            
            # Remove quotes if present
            category = category.strip('"\'')
            
            # Verify category exists in database
            if category in categories:
                return category
            else:
                print(f"Warning: Classified category '{category}' not in database.")
                print(f"Available categories: {categories}")
                # Try to find closest match (case-insensitive)
                category_lower = category.lower()
                for cat in categories:
                    if cat.lower() == category_lower:
                        print(f"Matched '{category}' to '{cat}' (case-insensitive)")
                        return cat
                return None
                
        except Exception as e:
            print(f"Error in classification: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _get_provider(self, provider_name: str, model_name: str, api_key: str) -> LLMProvider:
        """
        Get or create LLM provider instance (with caching).
        
        Args:
            provider_name: Provider name
            model_name: Model name
            api_key: API key
            
        Returns:
            LLMProvider instance
        """
        cache_key = f"{provider_name}:{model_name}"
        
        if cache_key not in self._provider_cache:
            self._provider_cache[cache_key] = create_provider(
                provider_name, model_name, api_key
            )
        
        return self._provider_cache[cache_key]
    
    def _normalize_model_label(self, label: str) -> str:
        """Normalize model label for matching against aliases."""
        normalized = label.strip().lower()
        normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized
    
    def _resolve_model_routing(self, model_label: str) -> Optional[Dict[str, str]]:
        """
        Resolve Supabase model label to OpenRouter model + API key env var.
        """
        if not model_label:
            return None
        
        normalized = self._normalize_model_label(model_label)
        
        # Exact alias match
        if normalized in self._model_routing_map:
            return self._model_routing_map[normalized]
        
        # Loose token matching
        if "kimi" in normalized and "k2" in normalized:
            return self._model_routing_map["kimi k2 thinking"]
        if "llama" in normalized and "405" in normalized:
            return self._model_routing_map["llama 3.51 405 b"]
        if "gpt" in normalized and "oss" in normalized:
            return self._model_routing_map["gpt oss 120 b"]
        if "nematron" in normalized or "nemotron" in normalized:
            return self._model_routing_map["nematron ultra 253b"]
        if "gemini" in normalized and "flash" in normalized:
            return self._model_routing_map["gemini flash 2.5 lite"]
        
        return None
    
    def process(self, user_text: str) -> str:
        """
        Process user text: classify, query Supabase, route to best LLM.
        
        Args:
            user_text: Text input from user (from STT)
            
        Returns:
            Response text from best LLM
        """
        try:
            # Step 1: Classify category using Gemini
            print(f"Classifying prompt category...")
            category = self._classify_category(user_text)
            
            if not category:
                # Fallback: use Gemini directly if classification fails
                print("Classification failed, using Gemini directly")
                response = self.router_client.chat.completions.create(
                    model=self.router_model,
                    messages=[{"role": "user", "content": user_text}]
                )
                return response.choices[0].message.content.strip()
            
            print(f"✓ Category: {category}")
            
            # Step 2: Query Supabase for best model
            print(f"Querying Supabase for best model...")
            model_config = self.benchmark_db.get_model_for_category(category)
            
            if not model_config:
                print(f"No model found for category '{category}', using Gemini directly")
                response = self.router_client.chat.completions.create(
                    model=self.router_model,
                    messages=[{"role": "user", "content": user_text}]
                )
                return response.choices[0].message.content.strip()
            
            model_label = model_config.get("model_name") or model_config.get("best_model")
            print(f"[debug] Supabase best_model: {model_label}")
            route = self._resolve_model_routing(model_label)
            provider_name = "openrouter"
            
            if not model_label:
                print(f"Incomplete model config (missing model name): {model_config}, using Gemini directly")
                response = self.router_client.chat.completions.create(
                    model=self.router_model,
                    messages=[{"role": "user", "content": user_text}]
                )
                return response.choices[0].message.content.strip()
            
            if not route:
                print(f"Unknown model label '{model_label}' (no routing match), using Gemini directly")
                response = self.router_client.chat.completions.create(
                    model=self.router_model,
                    messages=[{"role": "user", "content": user_text}]
                )
                return response.choices[0].message.content.strip()
            
            model_name = route["model"]
            api_key_env = route["api_key_env"]
            print(f"[debug] Routed model: {model_name} ({api_key_env})")
            
            # Step 3: Get API key from environment
            api_key = Settings.get_api_key(api_key_env)
            if not api_key:
                error_msg = f"API key not found for {api_key_env}. Please set it in .env file"
                print(f"Error: {error_msg}")
                # Fallback to Gemini
                print("Falling back to Gemini...")
                response = self.router_client.chat.completions.create(
                    model=self.router_model,
                    messages=[{"role": "user", "content": user_text}]
                )
                return response.choices[0].message.content.strip()
            
            print(f"✓ Routing to {provider_name} model: {model_name}")
            
            # Step 4: Get provider and generate response with original prompt
            provider = self._get_provider(provider_name, model_name, api_key)
            response = provider.generate(user_text)
            
            return response
            
        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            # Final fallback to Gemini
            try:
                print("Attempting fallback to Gemini...")
                response = self.router_client.chat.completions.create(
                    model=self.router_model,
                    messages=[{"role": "user", "content": user_text}]
                )
                return response.choices[0].message.content.strip()
            except:
                return error_msg
    
    def close(self):
        """Close all provider connections."""
        self._provider_cache.clear()
