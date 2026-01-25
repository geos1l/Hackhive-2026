"""Supabase client for querying benchmark database."""
from supabase import create_client, Client
from typing import Optional, Dict, List


class SupabaseBenchmarkDB:
    """Interface to Supabase benchmark database."""
    
    def __init__(self, supabase_url: str, supabase_key: str):
        """
        Initialize Supabase client.
        
        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase anon key or service role key
        """
        self.client: Client = create_client(supabase_url, supabase_key)
        self.table_name = "model_categories"  # Adjust to your table name if different
    
    def get_model_for_category(self, category: str) -> Optional[Dict]:
        """
        Query Supabase to get best model for a category/subject.
        
        Args:
            category: Category/subject name (e.g., "reasoning", "coding", etc.)
            
        Returns:
            Dict with model info or None if not found. Expected keys:
                "category_type", "best_model", "model_name",
                "provider", "api_key_env"
        """
        try:
            response = self.client.table(self.table_name)\
                .select("*")\
                .eq("category_type", category)\
                .limit(1)\
                .execute()
            
            if response.data and len(response.data) > 0:
                record = response.data[0]
                return {
                    "category_type": record.get("category_type"),
                    "best_model": record.get("best_model"),
                    "model_name": record.get("model_name"),
                    "provider": record.get("provider"),
                    "api_key_env": record.get("api_key_env"),
                }
            return None
        except Exception as e:
            print(f"Error querying Supabase: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_all_categories(self) -> List[str]:
        """
        Get list of all available categories from database.
        
        Returns:
            List of category names
        """
        try:
            response = self.client.table(self.table_name)\
                .select("category_type")\
                .execute()
            
            # Get unique categories
            categories = list(set([r.get("category_type") for r in response.data if r.get("category_type")]))
            return categories
        except Exception as e:
            print(f"Error fetching categories: {e}")
            import traceback
            traceback.print_exc()
            return []
