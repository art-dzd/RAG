"""Async OpenAI service with enhanced security and modern practices"""

import asyncio
import time
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config import settings
from app.utils.logging_config import get_logger
from app.utils.helpers import sanitize_text_input

logger = get_logger(__name__)


class OpenAIServiceError(Exception):
    """Custom exception for OpenAI service errors"""
    pass


class RateLimitError(OpenAIServiceError):
    """Exception for rate limit errors"""
    pass


class OpenAIService:
    """Async OpenAI service with enhanced features and security"""
    
    def __init__(self):
        """Initialize the OpenAI service"""
        try:
            self.client = AsyncOpenAI(
                api_key=settings.openai_api_key.get_secret_value(),
                timeout=settings.request_timeout,
                max_retries=3
            )
            
            # Model configuration
            self.embedding_model = settings.openai_embedding_model
            self.chat_model = settings.openai_model
            self.max_tokens = settings.openai_max_tokens
            self.temperature = settings.openai_temperature
            
            # Rate limiting
            self._request_count = 0
            self._last_reset = time.time()
            self._max_requests_per_minute = 60  # Conservative limit
            
            # Simple in-memory cache for embeddings
            self._embedding_cache = {}
            self._cache_max_size = 1000  # Maximum cache entries
            self._cache_ttl = 3600  # Cache TTL in seconds (1 hour)
            
            logger.info(f"OpenAI service initialized with model: {self.chat_model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI service: {e}")
            raise OpenAIServiceError(f"Initialization failed: {e}")
    
    async def _check_rate_limit(self) -> None:
        """Check and enforce rate limiting"""
        current_time = time.time()
        
        # Reset counter every minute
        if current_time - self._last_reset > 60:
            self._request_count = 0
            self._last_reset = current_time
        
        if self._request_count >= self._max_requests_per_minute:
            wait_time = 60 - (current_time - self._last_reset)
            if wait_time > 0:
                logger.warning(f"Rate limit exceeded, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
                self._request_count = 0
                self._last_reset = time.time()
        
        self._request_count += 1
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((Exception,))
    )
    async def create_embeddings(
        self, 
        texts: List[str], 
        batch_size: int = 50
    ) -> List[List[float]]:
        """
        Create embeddings for a list of texts with batching and retry logic
        
        Args:
            texts: List of texts to create embeddings for
            batch_size: Size of each batch for processing
            
        Returns:
            List of embedding vectors
            
        Raises:
            OpenAIServiceError: On API errors
            RateLimitError: On rate limit exceeded
        """
        if not texts:
            return []
        
        # Validate and sanitize inputs with size limits
        sanitized_texts = []
        total_tokens = 0
        max_tokens_per_text = 8192  # OpenAI limit for text-embedding-3-small
        
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                text = str(text)
            
            # Sanitize text input
            clean_text = sanitize_text_input(text, max_length=8000)
            if not clean_text.strip():
                clean_text = "empty"  # Fallback for empty texts
            
            # Check token count for this text
            estimated_tokens = len(clean_text.split()) * 1.3  # Rough estimation
            if estimated_tokens > max_tokens_per_text:
                logger.warning(f"Text {i} too long ({estimated_tokens:.0f} tokens), truncating")
                # Truncate to safe length
                words = clean_text.split()
                safe_words = words[:int(max_tokens_per_text / 1.3)]
                clean_text = " ".join(safe_words)
                estimated_tokens = len(safe_words) * 1.3
            
            total_tokens += estimated_tokens
            sanitized_texts.append(clean_text)
        
        # Check total batch size
        if total_tokens > 100000:  # Conservative limit for batch
            logger.warning(f"Batch too large ({total_tokens:.0f} tokens), reducing batch size")
            # Reduce batch size dynamically
            safe_batch_size = max(1, int(batch_size * (100000 / total_tokens)))
            batch_size = min(batch_size, safe_batch_size)
        
        all_embeddings = []
        
        try:
            # Process in batches
            for i in range(0, len(sanitized_texts), batch_size):
                batch = sanitized_texts[i:i + batch_size]
                
                await self._check_rate_limit()
                
                logger.debug(f"Creating embeddings for batch {i//batch_size + 1}, size: {len(batch)}")
                
                start_time = time.time()
                
                response = await self.client.embeddings.create(
                    model=self.embedding_model,
                    input=batch,
                    encoding_format="float"
                )
                
                # Extract embeddings from response
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                processing_time = time.time() - start_time
                logger.debug(f"Batch processed in {processing_time:.2f}s")
                
                # Small delay between batches to avoid overwhelming the API
                if i + batch_size < len(sanitized_texts):
                    await asyncio.sleep(0.1)
            
            # Cache embeddings
            for i, (text, embedding) in enumerate(zip(sanitized_texts, all_embeddings)):
                self._cache_embedding(text, embedding)
            
            logger.info(f"Created {len(all_embeddings)} embeddings successfully")
            return all_embeddings
            
        except Exception as e:
            error_msg = str(e)
            
            if "rate_limit" in error_msg.lower():
                logger.error("Rate limit exceeded")
                raise RateLimitError("OpenAI API rate limit exceeded")
            elif "quota" in error_msg.lower():
                logger.error("API quota exceeded")
                raise OpenAIServiceError("OpenAI API quota exceeded")
            else:
                logger.error(f"Error creating embeddings: {e}")
                raise OpenAIServiceError(f"Failed to create embeddings: {e}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        retry=retry_if_exception_type((Exception,))
    )
    async def create_single_embedding(self, text: str) -> List[float]:
        """
        Create embedding for a single text with retry logic
        
        Args:
            text: Text to create embedding for
            
        Returns:
            Embedding vector
            
        Raises:
            OpenAIServiceError: On API errors
        """
        if not text or not isinstance(text, str):
            text = str(text) if text else "empty"
        
        # Sanitize input with size limits
        clean_text = sanitize_text_input(text, max_length=8000)
        if not clean_text.strip():
            clean_text = "empty"
        
        # Check token count and truncate if necessary
        estimated_tokens = len(clean_text.split()) * 1.3
        max_tokens_per_text = 8192  # OpenAI limit for text-embedding-3-small
        
        if estimated_tokens > max_tokens_per_text:
            logger.warning(f"Text too long ({estimated_tokens:.0f} tokens), truncating")
            words = clean_text.split()
            safe_words = words[:int(max_tokens_per_text / 1.3)]
            clean_text = " ".join(safe_words)
        
        # Check cache first
        cached_embedding = self._get_cached_embedding(clean_text)
        if cached_embedding:
            return cached_embedding
        
        try:
            await self._check_rate_limit()
            
            response = await self.client.embeddings.create(
                model=self.embedding_model,
                input=clean_text,
                encoding_format="float"
            )
            
            embedding = response.data[0].embedding
            
            # Cache the embedding
            self._cache_embedding(clean_text, embedding)
            
            logger.debug(f"Created embedding for text length: {len(clean_text)}")
            return embedding
            
        except Exception as e:
            error_msg = str(e)
            
            if "rate_limit" in error_msg.lower():
                raise RateLimitError("OpenAI API rate limit exceeded")
            else:
                logger.error(f"Error creating single embedding: {e}")
                raise OpenAIServiceError(f"Failed to create embedding: {e}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,))
    )
    async def generate_chat_response(
        self,
        messages: List[Dict[str, str]],
        context_documents: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Generate chat response using the chat completion API
        
        Args:
            messages: List of messages in chat format
            context_documents: List of relevant documents for context
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            stream: Whether to stream the response
            
        Returns:
            Response dictionary with content and metadata
            
        Raises:
            OpenAIServiceError: On API errors
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")
        
        # Use instance defaults if not provided
        if max_tokens is None:
            max_tokens = self.max_tokens
        if temperature is None:
            temperature = self.temperature
        
        # Validate and sanitize messages
        clean_messages = []
        for msg in messages:
            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                logger.warning(f"Invalid message format: {msg}")
                continue
            
            role = msg['role']
            content = sanitize_text_input(str(msg['content']), max_length=4000)
            
            if role in ['user', 'assistant', 'system'] and content.strip():
                clean_messages.append({'role': role, 'content': content})
        
        if not clean_messages:
            raise ValueError("No valid messages after sanitization")
        
        try:
            await self._check_rate_limit()
            
            # Build system message with context
            system_message = self._build_system_message(context_documents)
            if system_message:
                # Insert system message at the beginning
                final_messages = [system_message] + clean_messages
            else:
                final_messages = clean_messages
            
            logger.debug(f"Sending {len(final_messages)} messages to {self.chat_model}")
            
            start_time = time.time()
            
            response = await self.client.chat.completions.create(
                model=self.chat_model,
                messages=final_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream,
                frequency_penalty=0.1,  # Reduce repetition
                presence_penalty=0.1   # Encourage topic diversity
            )
            
            response_time = time.time() - start_time
            
            # Extract response data
            if stream:
                # Handle streaming response (not implemented in this version)
                raise NotImplementedError("Streaming not implemented yet")
            else:
                choice = response.choices[0]
                content = choice.message.content
                usage = response.usage
                
                result = {
                    'content': content,
                    'role': 'assistant',
                    'usage': {
                        'prompt_tokens': usage.prompt_tokens,
                        'completion_tokens': usage.completion_tokens,
                        'total_tokens': usage.total_tokens
                    },
                    'response_time_ms': int(response_time * 1000),
                    'model': self.chat_model,
                    'finish_reason': choice.finish_reason
                }
                
                logger.info(
                    f"Generated response in {response_time:.2f}s, "
                    f"tokens: {usage.total_tokens}"
                )
                
                return result
                
        except Exception as e:
            error_msg = str(e)
            
            if "rate_limit" in error_msg.lower():
                raise RateLimitError("OpenAI API rate limit exceeded")
            elif "quota" in error_msg.lower():
                raise OpenAIServiceError("OpenAI API quota exceeded")
            elif "context_length" in error_msg.lower():
                raise OpenAIServiceError("Input too long for model context")
            else:
                logger.error(f"Error generating chat response: {e}")
                raise OpenAIServiceError(f"Failed to generate response: {e}")
    
    def _build_system_message(self, context_documents: Optional[List[str]]) -> Optional[Dict[str, str]]:
        """
        Build system message with context documents
        
        Args:
            context_documents: List of relevant documents
            
        Returns:
            System message dictionary or None
        """
        if not context_documents:
            return {
                'role': 'system',
                'content': (
                    "You are a helpful AI assistant that answers questions accurately and informatively. "
                    "If you don't know the answer, say so clearly. "
                    "Always respond in the same language as the user's question."
                )
            }
        
        # Combine and sanitize context documents
        clean_docs = []
        total_length = 0
        max_context_length = 6000  # Leave room for the actual conversation
        
        for doc in context_documents:
            if isinstance(doc, str) and doc.strip():
                clean_doc = sanitize_text_input(doc, max_length=2000)
                if clean_doc and total_length + len(clean_doc) < max_context_length:
                    clean_docs.append(clean_doc)
                    total_length += len(clean_doc)
                else:
                    break  # Stop if we exceed context length
        
        if not clean_docs:
            return self._build_system_message(None)
        
        context_text = "\n\n---\n\n".join(clean_docs)
        
        system_prompt = f"""You are an AI assistant that answers questions based on provided documents.

IMPORTANT RULES:
1. Answer ONLY based on information from the provided documents
2. If the documents don't contain enough information to answer, say so clearly
3. Don't make up information that isn't in the documents
4. Reference specific parts of documents when answering
5. Always respond in the same language as the user's question

CONTEXT FROM DOCUMENTS:
{context_text}

Answer user questions based on this context."""

        return {
            'role': 'system', 
            'content': system_prompt
        }
    
    async def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text (rough approximation)
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        if not text:
            return 0
        
        # Rough estimation: ~4 characters per token for English, ~3 for Russian
        # This is approximate - for exact count, we'd need tiktoken
        char_count = len(text)
        
        # Detect if text is primarily Russian (Cyrillic)
        cyrillic_chars = len([c for c in text if '\u0400' <= c <= '\u04FF'])
        if cyrillic_chars > char_count * 0.3:  # >30% Cyrillic
            return max(1, char_count // 3)
        else:
            return max(1, char_count // 4)
    
    def get_max_context_length(self) -> int:
        """Get maximum context length for the current model"""
        model_limits = {
            'gpt-4o-mini': 128000,
            'gpt-4o': 128000,
            'gpt-4-turbo': 128000,
            'gpt-4': 8192,
            'gpt-3.5-turbo': 16385,
        }
        
        return model_limits.get(self.chat_model, 4096)
    
    async def validate_api_key(self) -> bool:
        """
        Validate that the API key is working
        
        Returns:
            True if API key is valid
        """
        try:
            # Test with a minimal request
            await self.create_single_embedding("test")
            return True
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return False
    
    async def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about available models
        
        Returns:
            Dictionary with model information
        """
        try:
            models = await self.client.models.list()
            
            available_models = []
            for model in models.data:
                if any(prefix in model.id for prefix in ['gpt', 'text-embedding']):
                    available_models.append({
                        'id': model.id,
                        'created': model.created,
                        'owned_by': model.owned_by
                    })
            
            return {
                'available_models': available_models,
                'current_chat_model': self.chat_model,
                'current_embedding_model': self.embedding_model
            }
            
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {'error': str(e)}
    
    def _cache_embedding(self, text: str, embedding: List[float]) -> None:
        """Cache an embedding with TTL"""
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Clean cache if it's too large
        if len(self._embedding_cache) >= self._cache_max_size:
            # Remove oldest entries
            current_time = time.time()
            self._embedding_cache = {
                k: v for k, v in self._embedding_cache.items()
                if current_time - v['timestamp'] < self._cache_ttl
            }
        
        self._embedding_cache[text_hash] = {
            'embedding': embedding,
            'timestamp': time.time()
        }
    
    def _get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding if available and not expired"""
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash in self._embedding_cache:
            cached = self._embedding_cache[text_hash]
            if time.time() - cached['timestamp'] < self._cache_ttl:
                logger.debug(f"Cache hit for text hash: {text_hash[:8]}")
                return cached['embedding']
            else:
                # Remove expired entry
                del self._embedding_cache[text_hash]
        
        return None
    
    def clear_cache(self) -> None:
        """Clear the embedding cache"""
        self._embedding_cache.clear()
        logger.info("Embedding cache cleared")
    
    async def close(self):
        """Close the HTTP client"""
        if hasattr(self.client, '_client'):
            await self.client._client.aclose()


# Global service instance
openai_service = OpenAIService() 