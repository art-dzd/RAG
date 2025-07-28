"""Основной файл для запуска RAG Telegram Bot системы"""

import asyncio
import signal
import sys
from typing import Optional
import uvicorn
from concurrent.futures import ThreadPoolExecutor

from app.config import settings
from app.utils.logging_config import setup_logging, get_logger
from app.telegram.bot import run_bot
from app.database.database import create_tables


# Настройка логирования
setup_logging(settings.log_level, settings.log_file)
logger = get_logger(__name__)


class RAGBotSystem:
    """Основной класс системы RAG бота"""
    
    def __init__(self):
        self.fastapi_server = None
        self.bot_task = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.shutdown_event = asyncio.Event()
        
        logger.info("RAG Bot система инициализирована")
    
    async def start_fastapi_server(self):
        """Запустить FastAPI сервер"""
        try:
            logger.info(f"Запуск FastAPI сервера на {settings.api_host}:{settings.api_port}")
            
            config = uvicorn.Config(
                "app.main:app",
                host=settings.api_host,
                port=settings.api_port,
                log_level=settings.log_level.lower(),
                reload=False,
                access_log=True
            )
            
            server = uvicorn.Server(config)
            await server.serve()
            
        except Exception as e:
            logger.error(f"Ошибка запуска FastAPI сервера: {e}")
            raise
    
    async def start_telegram_bot(self):
        """Запустить Telegram бота"""
        try:
            logger.info("Запуск Telegram бота")
            await run_bot()
        except Exception as e:
            logger.error(f"Ошибка запуска Telegram бота: {e}")
            raise
    
    async def initialize_database(self):
        """Инициализировать базу данных"""
        try:
            logger.info("Инициализация базы данных")
            create_tables()
            logger.info("База данных инициализирована")
        except Exception as e:
            logger.error(f"Ошибка инициализации базы данных: {e}")
            raise
    
    async def start_system(self):
        """Запустить всю систему"""
        try:
            logger.info("=== Запуск RAG Telegram Bot системы ===")
            
            # 1. Инициализировать базу данных
            await self.initialize_database()
            
            # 2. Создать задачи для FastAPI и Telegram бота
            fastapi_task = asyncio.create_task(
                self.start_fastapi_server(),
                name="FastAPI-Server"
            )
            
            bot_task = asyncio.create_task(
                self.start_telegram_bot(),
                name="Telegram-Bot"
            )
            
            self.fastapi_server = fastapi_task
            self.bot_task = bot_task
            
            logger.info("Все сервисы запущены успешно")
            logger.info(f"FastAPI доступен на: http://{settings.api_host}:{settings.api_port}")
            logger.info("Telegram бот готов к работе")
            
            # 3. Ожидать завершения любой из задач или сигнала остановки
            done, pending = await asyncio.wait(
                [fastapi_task, bot_task, asyncio.create_task(self.shutdown_event.wait())],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Если одна из задач завершилась с ошибкой
            for task in done:
                if task.get_name() != "Task-3" and task.exception():  # Task-3 это shutdown_event
                    logger.error(f"Задача {task.get_name()} завершилась с ошибкой: {task.exception()}")
                    await self.shutdown_system()
                    break
            
        except KeyboardInterrupt:
            logger.info("Получен сигнал прерывания")
            await self.shutdown_system()
        except Exception as e:
            logger.error(f"Критическая ошибка системы: {e}")
            await self.shutdown_system()
            raise
    
    async def shutdown_system(self):
        """Корректно завершить работу системы"""
        try:
            logger.info("Начало корректного завершения системы...")
            
            # Сигнализировать о завершении
            self.shutdown_event.set()
            
            # Отменить задачи
            tasks_to_cancel = []
            
            if self.fastapi_server and not self.fastapi_server.done():
                tasks_to_cancel.append(self.fastapi_server)
            
            if self.bot_task and not self.bot_task.done():
                tasks_to_cancel.append(self.bot_task)
            
            if tasks_to_cancel:
                logger.info(f"Отмена {len(tasks_to_cancel)} задач...")
                
                for task in tasks_to_cancel:
                    task.cancel()
                
                # Ожидать завершения с таймаутом
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*tasks_to_cancel, return_exceptions=True),
                        timeout=10.0
                    )
                except asyncio.TimeoutError:
                    logger.warning("Таймаут при завершении задач")
            
            # Закрыть executor
            self.executor.shutdown(wait=True)
            
            logger.info("Система корректно завершена")
            
        except Exception as e:
            logger.error(f"Ошибка при завершении системы: {e}")
    
    def setup_signal_handlers(self):
        """Настроить обработчики сигналов"""
        def signal_handler(signum, frame):
            logger.info(f"Получен сигнал {signum}")
            asyncio.create_task(self.shutdown_system())
        
        # Настроить обработчики для Unix-систем
        if sys.platform != "win32":
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)


async def main():
    """Главная функция"""
    system = RAGBotSystem()
    
    try:
        # Настроить обработчики сигналов
        system.setup_signal_handlers()
        
        # Запустить систему
        await system.start_system()
        
    except KeyboardInterrupt:
        logger.info("Прерывание пользователем")
    except Exception as e:
        logger.error(f"Фатальная ошибка: {e}")
        return 1
    
    return 0


def validate_configuration():
    """Валидировать конфигурацию перед запуском"""
    errors = []
    
    # Проверить обязательные переменные окружения
    if not settings.openai_api_key or settings.openai_api_key == "your_openai_api_key_here":
        errors.append("OPENAI_API_KEY не установлен или имеет значение по умолчанию")
    
    if not settings.telegram_bot_token or settings.telegram_bot_token == "your_telegram_bot_token_here":
        errors.append("TELEGRAM_BOT_TOKEN не установлен или имеет значение по умолчанию")
    
    # Проверить числовые значения
    if settings.api_port < 1 or settings.api_port > 65535:
        errors.append(f"Неверный порт API: {settings.api_port}")
    
    if settings.max_file_size_mb <= 0:
        errors.append(f"Неверный максимальный размер файла: {settings.max_file_size_mb}")
    
    if errors:
        logger.error("Ошибки конфигурации:")
        for error in errors:
            logger.error(f"  - {error}")
        logger.error("Пожалуйста, исправьте конфигурацию в файле .env")
        return False
    
    return True


if __name__ == "__main__":
    print("🤖 RAG Telegram Bot System")
    print("=" * 50)
    
    # Валидировать конфигурацию
    if not validate_configuration():
        sys.exit(1)
    
    # Показать конфигурацию
    print(f"📡 API сервер: http://{settings.api_host}:{settings.api_port}")
    print(f"📁 База данных: {settings.database_url}")
    print(f"🗂 Векторная БД: {settings.chroma_db_path}")
    print(f"📋 Логи: {settings.log_file}")
    print(f"🔧 Режим отладки: {'Включён' if settings.debug else 'Выключен'}")
    print("=" * 50)
    
    try:
        # Запустить систему
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n👋 Система остановлена пользователем")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Критическая ошибка: {e}")
        sys.exit(1) 