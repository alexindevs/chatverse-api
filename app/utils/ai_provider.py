"""
AI Provider Abstraction Layer
Supports both OpenAI and Google Gemini APIs with a unified interface.
"""
import os
import logging
from typing import List, Dict, Any, Optional
from enum import Enum
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class AIProvider(str, Enum):
    """Supported AI providers"""
    OPENAI = "openai"
    GEMINI = "gemini"


class AIProviderClient:
    """
    Unified interface for AI providers (OpenAI and Gemini).
    Automatically selects provider based on environment variable.
    """
    
    def __init__(self):
        self.provider = os.getenv("AI_PROVIDER", "openai").lower()
        
        if self.provider == AIProvider.OPENAI:
            self._init_openai()
        elif self.provider == AIProvider.GEMINI:
            self._init_gemini()
        else:
            raise ValueError(f"Unsupported AI provider: {self.provider}. Use 'openai' or 'gemini'")
    
    def _init_openai(self):
        """Initialize OpenAI client"""
        try:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required")
            self.client = OpenAI(api_key=api_key)
            self._chat_completion = self._openai_chat_completion
            self._create_embedding = self._openai_create_embedding
            logger.info("Initialized OpenAI client")
        except ImportError:
            raise ImportError("openai package is required. Install with: pip install openai")
    
    def _init_gemini(self):
        """Initialize Gemini client"""
        try:
            import google.generativeai as genai
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable is required")
            genai.configure(api_key=api_key)
            self.client = genai
            self._chat_completion = self._gemini_chat_completion
            self._create_embedding = self._gemini_create_embedding
            logger.info("Initialized Gemini client")
        except ImportError:
            raise ImportError("google-generativeai package is required. Install with: pip install google-generativeai")
    
    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7
    ) -> str:
        """
        Generate chat completion from messages.
        
        Args:
            model: Model name (e.g., "gpt-4o" or "gemini-pro")
            messages: List of message dicts with "role" and "content" keys
            temperature: Sampling temperature (0.0 to 1.0)
        
        Returns:
            Generated text content
        """
        return await self._chat_completion(model, messages, temperature)
    
    async def _openai_chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float
    ) -> str:
        """OpenAI chat completion implementation"""
        import asyncio
        
        # Run synchronous OpenAI calls in thread pool to make them async-compatible
        def _sync_openai_call():
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content
        
        # Execute in thread pool to avoid blocking
        return await asyncio.to_thread(_sync_openai_call)
    
    async def _gemini_chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float
    ) -> str:
        """Gemini chat completion implementation"""
        import asyncio
        
        # Convert OpenAI message format to Gemini format
        # Gemini doesn't support system messages directly, so we prepend it to the first user message
        gemini_messages = []
        system_content = None
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                if system_content:
                    system_content += "\n\n" + content
                else:
                    system_content = content
            elif role == "user":
                if system_content:
                    # Prepend system message to first user message
                    full_content = system_content + "\n\n" + content
                    gemini_messages.append({"role": "user", "parts": [full_content]})
                    system_content = None  # Only prepend once
                else:
                    gemini_messages.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                gemini_messages.append({"role": "model", "parts": [content]})
        
        # If system content remains, add it to the last user message
        if system_content and gemini_messages:
            last_msg = gemini_messages[-1]
            if last_msg["role"] == "user":
                last_msg["parts"][0] = system_content + "\n\n" + last_msg["parts"][0]
        
        # Initialize the model
        genai_model = self.client.GenerativeModel(
            model_name=model,
            generation_config={
                "temperature": temperature,
            }
        )
        
        # Run synchronous Gemini calls in thread pool to make them async-compatible
        def _sync_gemini_call():
            # Start a chat session if we have history
            if len(gemini_messages) > 1:
                chat = genai_model.start_chat(history=gemini_messages[:-1])
                response = chat.send_message(gemini_messages[-1]["parts"][0])
            else:
                # Single message, no history
                prompt = gemini_messages[0]["parts"][0] if gemini_messages else ""
                response = genai_model.generate_content(prompt)
            return response.text
        
        # Execute in thread pool to avoid blocking
        response_text = await asyncio.to_thread(_sync_gemini_call)
        return response_text
    
    async def create_embedding(
        self,
        model: str,
        input_text: str
    ) -> List[float]:
        """
        Create embedding vector for input text.
        
        Args:
            model: Embedding model name
            input_text: Text to embed
        
        Returns:
            List of float values representing the embedding
        """
        return await self._create_embedding(model, input_text)
    
    async def _openai_create_embedding(
        self,
        model: str,
        input_text: str
    ) -> List[float]:
        """OpenAI embedding implementation"""
        import asyncio
        
        # Run synchronous OpenAI calls in thread pool to make them async-compatible
        def _sync_openai_call():
            response = self.client.embeddings.create(
                model=model,
                input=input_text
            )
            return response.data[0].embedding
        
        # Execute in thread pool to avoid blocking
        return await asyncio.to_thread(_sync_openai_call)
    
    async def _gemini_create_embedding(
        self,
        model: str,
        input_text: str
    ) -> List[float]:
        """Gemini embedding implementation"""
        import asyncio
        
        # Run synchronous Gemini embedding calls in thread pool
        def _sync_embedding_call():
            try:
                result = self.client.embed_content(
                    model=model,
                    content=input_text,
                    task_type="retrieval_document"  # or "retrieval_query", "semantic_similarity", etc.
                )
                return result['embedding']
            except Exception as e:
                # Fallback: try alternative method
                logger.warning(f"Primary embedding method failed: {e}, trying alternative")
                result = self.client.embed_content(
                    model=model,
                    content=input_text
                )
                return result['embedding']
        
        # Execute in thread pool to avoid blocking
        return await asyncio.to_thread(_sync_embedding_call)
    
    def get_default_chat_model(self) -> str:
        """Get the default chat model for the current provider"""
        if self.provider == AIProvider.OPENAI:
            return os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o")
        elif self.provider == AIProvider.GEMINI:
            return os.getenv("GEMINI_DEFAULT_MODEL", "gemini-pro")
        return "gpt-4o"
    
    def get_default_embedding_model(self) -> str:
        """Get the default embedding model for the current provider"""
        if self.provider == AIProvider.OPENAI:
            return os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
        elif self.provider == AIProvider.GEMINI:
            return os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001")
        return "text-embedding-ada-002"
    
    def handle_error(self, error: Exception) -> tuple[int, str]:
        """
        Handle provider-specific errors and return appropriate HTTP status and message.
        
        Returns:
            Tuple of (status_code, error_message)
        """
        error_str = str(error)
        
        if self.provider == AIProvider.OPENAI:
            import openai
            if isinstance(error, openai.BadRequestError):
                return (400, f"Invalid request to AI provider: {error_str}")
            elif isinstance(error, (openai.APIError, openai.APIConnectionError, openai.RateLimitError)):
                return (503, f"AI provider unavailable: {error_str}")
        elif self.provider == AIProvider.GEMINI:
            # Gemini error handling
            if "400" in error_str or "BadRequest" in error_str or "invalid" in error_str.lower():
                return (400, f"Invalid request to AI provider: {error_str}")
            elif "429" in error_str or "rate limit" in error_str.lower() or "quota" in error_str.lower():
                return (503, f"AI provider rate limit exceeded: {error_str}")
            elif "503" in error_str or "unavailable" in error_str.lower():
                return (503, f"AI provider unavailable: {error_str}")
        
        # Generic error
        return (503, f"Error communicating with AI provider: {error_str}")


# Global instance - initialized on first import
_ai_client: Optional[AIProviderClient] = None


def get_ai_client() -> AIProviderClient:
    """Get or create the global AI client instance"""
    global _ai_client
    if _ai_client is None:
        _ai_client = AIProviderClient()
    return _ai_client
