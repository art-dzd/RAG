"""Основной файл Telegram бота"""

import asyncio
import sys
from typing import Any

from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram.fsm.storage.memory import MemoryStorage

from app.config import settings
from app.telegram.handlers import router
from app.utils.logging_config import setup_logging, get_logger

# Настройка логирования
setup_logging(settings.log_level, settings.log_file)
logger = get_logger(__name__)


class TelegramBot:
    """Класс Telegram бота"""
    
    def __init__(self):
        """Инициализация бота"""
        self.bot = Bot(
            token=settings.telegram_bot_token.get_secret_value(),
            default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN)
        )
        
        # Хранилище состояний в памяти
        self.storage = MemoryStorage()
        
        # Диспетчер
        self.dp = Dispatcher(storage=self.storage)
        
        # Регистрация роутеров
        self.dp.include_router(router)
        
        logger.info("Telegram бот инициализирован")
    
    async def start_polling(self):
        """Запустить бота в режиме long polling"""
        try:
            logger.info("Запуск Telegram бота...")
            
            # Удалить webhook если он был установлен
            await self.bot.delete_webhook(drop_pending_updates=True)
            
            # Получить информацию о боте
            bot_info = await self.bot.get_me()
            logger.info(f"Бот запущен: @{bot_info.username} ({bot_info.full_name})")
            
            # Запустить polling
            await self.dp.start_polling(
                self.bot,
                polling_timeout=20,
                handle_as_tasks=True
            )
            
        except Exception as e:
            logger.error(f"Ошибка при запуске бота: {e}")
            raise
    
    async def stop(self):
        """Остановить бота"""
        try:
            logger.info("Остановка Telegram бота...")
            
            # Закрыть сессию бота
            await self.bot.session.close()
            
            # Остановить диспетчер
            await self.dp.stop_polling()
            
            logger.info("Telegram бот остановлен")
            
        except Exception as e:
            logger.error(f"Ошибка при остановке бота: {e}")
    
    async def send_message(self, chat_id: int, text: str, **kwargs) -> Any:
        """
        Отправить сообщение
        
        Args:
            chat_id: ID чата
            text: Текст сообщения
            **kwargs: Дополнительные параметры
            
        Returns:
            Отправленное сообщение
        """
        try:
            return await self.bot.send_message(chat_id, text, **kwargs)
        except Exception as e:
            logger.error(f"Ошибка отправки сообщения: {e}")
            raise
    
    async def send_document(self, chat_id: int, document: Any, **kwargs) -> Any:
        """
        Отправить документ
        
        Args:
            chat_id: ID чата
            document: Документ
            **kwargs: Дополнительные параметры
            
        Returns:
            Отправленное сообщение
        """
        try:
            return await self.bot.send_document(chat_id, document, **kwargs)
        except Exception as e:
            logger.error(f"Ошибка отправки документа: {e}")
            raise


# Глобальный экземпляр бота
telegram_bot = None


async def run_bot():
    """Запустить бота"""
    global telegram_bot
    
    # Создаем экземпляр бота только при запуске
    telegram_bot = TelegramBot()
    
    try:
        await telegram_bot.start_polling()
    except KeyboardInterrupt:
        logger.info("Получен сигнал остановки")
    except Exception as e:
        logger.error(f"Критическая ошибка бота: {e}")
        sys.exit(1)
    finally:
        if telegram_bot:
            await telegram_bot.stop()


if __name__ == "__main__":
    # Запуск бота
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        logger.info("Бот остановлен пользователем")
    except Exception as e:
        logger.error(f"Фатальная ошибка: {e}")
        sys.exit(1) 