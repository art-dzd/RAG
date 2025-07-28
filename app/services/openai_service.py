"""Сервис для работы с OpenAI API"""

import time
from typing import List, Dict, Any, Optional
import openai
from openai import OpenAI

from app.config import settings
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class OpenAIServiceError(Exception):
    """Исключение для ошибок OpenAI сервиса"""
    pass


class OpenAIService:
    """Сервис для работы с OpenAI API"""
    
    def __init__(self):
        """Инициализация сервиса"""
        try:
            self.client = OpenAI(api_key=settings.openai_api_key)
            self.embedding_model = "text-embedding-3-small"  # Более новая и эффективная модель
            self.chat_model = "gpt-4-turbo-preview"  # Обновленная модель
            
            # Проверка соединения с API
            self._test_connection()
            
            logger.info("OpenAI сервис инициализирован")
        except Exception as e:
            logger.error(f"Ошибка инициализации OpenAI сервиса: {e}")
            raise OpenAIServiceError(f"Не удалось инициализировать OpenAI сервис: {e}")
    
    def _test_connection(self):
        """Проверить соединение с OpenAI API"""
        try:
            # Простой тест с минимальным запросом
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input="test"
            )
            logger.info("Соединение с OpenAI API успешно установлено")
        except Exception as e:
            logger.error(f"Ошибка соединения с OpenAI API: {e}")
            raise OpenAIServiceError(f"Не удалось подключиться к OpenAI API: {e}")
    
    async def create_embeddings(
        self, 
        texts: List[str], 
        batch_size: int = 100
    ) -> List[List[float]]:
        """
        Создать эмбеддинги для списка текстов
        
        Args:
            texts: Список текстов для создания эмбеддингов
            batch_size: Размер батча для обработки
            
        Returns:
            Список векторов эмбеддингов
            
        Raises:
            OpenAIServiceError: При ошибке создания эмбеддингов
        """
        try:
            all_embeddings = []
            
            # Обработка батчами для больших объёмов данных
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                logger.info(f"Обработка батча {i//batch_size + 1}, размер: {len(batch)}")
                
                start_time = time.time()
                
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=batch
                )
                
                # Извлечь эмбеддинги из ответа
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                processing_time = time.time() - start_time
                logger.info(f"Батч обработан за {processing_time:.2f} секунд")
                
                # Небольшая пауза между батчами для соблюдения лимитов API
                if i + batch_size < len(texts):
                    time.sleep(0.1)
            
            logger.info(f"Создано {len(all_embeddings)} эмбеддингов")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Ошибка создания эмбеддингов: {e}")
            raise OpenAIServiceError(f"Не удалось создать эмбеддинги: {e}")
    
    async def create_single_embedding(self, text: str) -> List[float]:
        """
        Создать эмбеддинг для одного текста
        
        Args:
            text: Текст для создания эмбеддинга
            
        Returns:
            Вектор эмбеддинга
            
        Raises:
            OpenAIServiceError: При ошибке создания эмбеддинга
        """
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            
            embedding = response.data[0].embedding
            logger.debug(f"Создан эмбеддинг для текста длиной {len(text)} символов")
            return embedding
            
        except Exception as e:
            logger.error(f"Ошибка создания эмбеддинга: {e}")
            raise OpenAIServiceError(f"Не удалось создать эмбеддинг: {e}")
    
    async def generate_chat_response(
        self,
        messages: List[Dict[str, str]],
        context_documents: Optional[List[str]] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Генерировать ответ используя Chat API
        
        Args:
            messages: История сообщений в формате [{"role": "user", "content": "..."}]
            context_documents: Список релевантных документов для контекста
            max_tokens: Максимальное количество токенов в ответе
            temperature: Температура генерации (0.0 - 2.0)
            
        Returns:
            Словарь с ответом и метаданными
            
        Raises:
            OpenAIServiceError: При ошибке генерации ответа
        """
        try:
            start_time = time.time()
            
            # Подготовить сообщения
            system_message = self._build_system_message(context_documents)
            full_messages = [system_message] + messages if system_message else messages
            
            logger.info(f"Отправка запроса к GPT-4 с {len(full_messages)} сообщениями")
            
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=full_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False
            )
            
            response_time = time.time() - start_time
            
            # Извлечь данные из ответа
            assistant_message = response.choices[0].message.content
            usage = response.usage
            
            result = {
                'content': assistant_message,
                'role': 'assistant',
                'usage': {
                    'prompt_tokens': usage.prompt_tokens,
                    'completion_tokens': usage.completion_tokens,
                    'total_tokens': usage.total_tokens
                },
                'response_time_ms': int(response_time * 1000),
                'model': self.chat_model
            }
            
            logger.info(
                f"Ответ получен за {response_time:.2f}с, "
                f"токенов: {usage.total_tokens}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка генерации ответа: {e}")
            raise OpenAIServiceError(f"Не удалось сгенерировать ответ: {e}")
    
    def _build_system_message(self, context_documents: Optional[List[str]]) -> Optional[Dict[str, str]]:
        """
        Построить системное сообщение с контекстом
        
        Args:
            context_documents: Список релевантных документов
            
        Returns:
            Системное сообщение или None
        """
        if not context_documents:
            return {
                'role': 'system',
                'content': (
                    "Ты - полезный AI-ассистент, который отвечает на вопросы пользователей. "
                    "Отвечай точно, информативно и по существу. Если не знаешь ответа, так и скажи."
                )
            }
        
        # Объединить документы в контекст
        context_text = "\n\n---\n\n".join(context_documents)
        
        system_prompt = f"""Ты - AI-ассистент, который отвечает на вопросы на основе предоставленных документов.

ВАЖНЫЕ ПРАВИЛА:
1. Отвечай ТОЛЬКО на основе информации из предоставленных документов
2. Если информации в документах недостаточно для ответа, честно скажи об этом
3. Не придумывай информацию, которой нет в документах
4. Ссылайся на конкретные части документов при ответе
5. Отвечай на русском языке

КОНТЕКСТ ИЗ ДОКУМЕНТОВ:
{context_text}

Отвечай на вопросы пользователя на основе этого контекста."""

        return {
            'role': 'system', 
            'content': system_prompt
        }
    
    async def estimate_tokens(self, text: str) -> int:
        """
        Оценить количество токенов в тексте
        
        Args:
            text: Текст для оценки
            
        Returns:
            Приблизительное количество токенов
        """
        # Простая оценка: ~4 символа = 1 токен для русского текста
        # Это приблизительная оценка, для точной нужна tiktoken библиотека
        return max(1, len(text) // 4)
    
    def get_max_context_length(self) -> int:
        """Получить максимальную длину контекста для модели"""
        # GPT-4 Turbo имеет контекст 128k токенов
        return 128000


# Глобальный экземпляр сервиса
openai_service = OpenAIService() 