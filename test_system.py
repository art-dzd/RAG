#!/usr/bin/env python3
"""
Скрипт для тестирования всех функций RAG Telegram Bot системы
"""

import asyncio
import os
import sys
import json
import tempfile
from pathlib import Path

# Добавить корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent))

import httpx
from app.config import settings
from app.utils.logging_config import setup_logging, get_logger

# Настройка логирования для тестов
setup_logging("INFO", "./logs/test.log")
logger = get_logger(__name__)


class SystemTester:
    """Класс для тестирования системы"""
    
    def __init__(self):
        self.api_base = f"http://{settings.api_host}:{settings.api_port}"
        self.test_user_id = "test_user_123"
        self.client = httpx.AsyncClient(timeout=60.0)
        
    async def test_api_health(self):
        """Тест здоровья API"""
        try:
            logger.info("🏥 Тестирование health check...")
            response = await self.client.get(f"{self.api_base}/health")
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Health check пройден: {data['status']}")
                return True
            else:
                logger.error(f"❌ Health check не пройден: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Ошибка при тестировании health check: {e}")
            return False
    
    async def test_root_endpoint(self):
        """Тест корневого эндпоинта"""
        try:
            logger.info("🏠 Тестирование корневого эндпоинта...")
            response = await self.client.get(f"{self.api_base}/")
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Корневой эндпоинт работает: {data['message']}")
                return True
            else:
                logger.error(f"❌ Корневой эндпоинт не работает: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Ошибка при тестировании корневого эндпоинта: {e}")
            return False
    
    async def test_user_creation(self):
        """Тест создания пользователя"""
        try:
            logger.info("👤 Тестирование создания пользователя...")
            
            user_data = {
                "telegram_id": self.test_user_id,
                "username": "test_user",
                "first_name": "Test",
                "last_name": "User"
            }
            
            response = await self.client.post(f"{self.api_base}/users/", json=user_data)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Пользователь создан: ID {data['id']}")
                return True, data
            else:
                logger.error(f"❌ Ошибка создания пользователя: {response.status_code}")
                return False, None
                
        except Exception as e:
            logger.error(f"❌ Ошибка при тестировании создания пользователя: {e}")
            return False, None
    
    async def test_document_upload(self):
        """Тест загрузки документа"""
        try:
            logger.info("📄 Тестирование загрузки документа...")
            
            # Создать тестовый документ
            test_content = """
            Тестовый документ для RAG системы
            
            Этот документ содержит информацию о тестировании системы.
            Система должна быть способна отвечать на вопросы по этому документу.
            
            Основные функции:
            1. Обработка документов
            2. Создание эмбеддингов
            3. Поиск релевантной информации
            4. Генерация ответов
            
            Тестовая информация:
            - Название системы: RAG Telegram Bot
            - Версия: 1.0.0
            - Разработчик: AI Assistant
            """
            
            # Сохранить во временный файл
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
                f.write(test_content)
                temp_file_path = f.name
            
            try:
                # Загрузить файл
                with open(temp_file_path, 'rb') as f:
                    files = {"file": ("test_document.txt", f, "text/plain")}
                    data = {"user_id": self.test_user_id}
                    
                    response = await self.client.post(
                        f"{self.api_base}/upload/", 
                        files=files, 
                        data=data
                    )
                
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"✅ Документ загружен: ID {data['document_id']}, чанков: {data['chunks_count']}")
                    return True, data['document_id']
                else:
                    logger.error(f"❌ Ошибка загрузки документа: {response.status_code} - {response.text}")
                    return False, None
                    
            finally:
                # Удалить временный файл
                os.unlink(temp_file_path)
                
        except Exception as e:
            logger.error(f"❌ Ошибка при тестировании загрузки документа: {e}")
            return False, None
    
    async def test_document_query(self, document_id):
        """Тест запроса к документу"""
        try:
            logger.info("💬 Тестирование запроса к документу...")
            
            test_queries = [
                "О чём этот документ?",
                "Какие функции есть в системе?",
                "Какая версия системы?",
                "Кто разработчик?"
            ]
            
            successful_queries = 0
            
            for query in test_queries:
                query_data = {
                    "user_id": self.test_user_id,
                    "document_id": document_id,
                    "query": query,
                    "chat_history": []
                }
                
                response = await self.client.post(f"{self.api_base}/query/", json=query_data)
                
                if response.status_code == 200:
                    data = response.json()
                    if data['success']:
                        logger.info(f"✅ Запрос '{query}' успешен. Найдено чанков: {data['found_chunks']}")
                        logger.info(f"   Ответ: {data['answer'][:100]}...")
                        successful_queries += 1
                    else:
                        logger.warning(f"⚠️ Запрос '{query}' не дал результата: {data.get('error')}")
                else:
                    logger.error(f"❌ Ошибка запроса '{query}': {response.status_code}")
            
            success_rate = successful_queries / len(test_queries)
            logger.info(f"📊 Успешность запросов: {successful_queries}/{len(test_queries)} ({success_rate*100:.1f}%)")
            
            return success_rate > 0.5  # Считаем успешным если больше 50% запросов работают
            
        except Exception as e:
            logger.error(f"❌ Ошибка при тестировании запросов: {e}")
            return False
    
    async def test_user_documents_list(self):
        """Тест получения списка документов пользователя"""
        try:
            logger.info("📚 Тестирование списка документов...")
            
            response = await self.client.get(f"{self.api_base}/users/{self.test_user_id}/documents/")
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Список документов получен: {data['total_documents']} документов")
                return True
            else:
                logger.error(f"❌ Ошибка получения списка документов: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Ошибка при тестировании списка документов: {e}")
            return False
    
    async def test_metrics(self):
        """Тест получения метрик"""
        try:
            logger.info("📊 Тестирование метрик...")
            
            response = await self.client.get(f"{self.api_base}/metrics")
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Метрики получены:")
                logger.info(f"   Пользователей: {data['total_metrics']['users']}")
                logger.info(f"   Документов: {data['total_metrics']['documents']}")
                logger.info(f"   Сообщений: {data['total_metrics']['messages']}")
                return True
            else:
                logger.error(f"❌ Ошибка получения метрик: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Ошибка при тестировании метрик: {e}")
            return False
    
    async def run_all_tests(self):
        """Запустить все тесты"""
        logger.info("🚀 Начинаю полное тестирование системы...")
        
        test_results = {}
        
        # Тест 1: Health check
        test_results['health'] = await self.test_api_health()
        
        # Тест 2: Корневой эндпоинт
        test_results['root'] = await self.test_root_endpoint()
        
        # Тест 3: Создание пользователя
        user_success, user_data = await self.test_user_creation()
        test_results['user_creation'] = user_success
        
        # Тест 4: Загрузка документа (только если пользователь создан)
        document_id = None
        if user_success:
            doc_success, document_id = await self.test_document_upload()
            test_results['document_upload'] = doc_success
            
            # Тест 5: Запросы к документу (только если документ загружен)
            if doc_success and document_id:
                test_results['document_query'] = await self.test_document_query(document_id)
            else:
                test_results['document_query'] = False
            
            # Тест 6: Список документов
            test_results['documents_list'] = await self.test_user_documents_list()
        else:
            test_results['document_upload'] = False
            test_results['document_query'] = False
            test_results['documents_list'] = False
        
        # Тест 7: Метрики
        test_results['metrics'] = await self.test_metrics()
        
        # Подведение итогов
        await self.print_test_summary(test_results)
        
        return test_results
    
    async def print_test_summary(self, results):
        """Вывести итоги тестирования"""
        logger.info("\n" + "="*60)
        logger.info("📋 ИТОГИ ТЕСТИРОВАНИЯ")
        logger.info("="*60)
        
        total_tests = len(results)
        passed_tests = sum(1 for result in results.values() if result)
        
        for test_name, result in results.items():
            status = "✅ ПРОЙДЕН" if result else "❌ НЕ ПРОЙДЕН"
            logger.info(f"{test_name.upper().replace('_', ' ')}: {status}")
        
        logger.info("-"*60)
        logger.info(f"ВСЕГО ТЕСТОВ: {total_tests}")
        logger.info(f"ПРОЙДЕНО: {passed_tests}")
        logger.info(f"НЕ ПРОЙДЕНО: {total_tests - passed_tests}")
        logger.info(f"УСПЕШНОСТЬ: {passed_tests/total_tests*100:.1f}%")
        
        if passed_tests == total_tests:
            logger.info("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        elif passed_tests >= total_tests * 0.8:
            logger.info("⚠️ БОЛЬШИНСТВО ТЕСТОВ ПРОЙДЕНО")
        else:
            logger.info("❌ МНОГО ТЕСТОВ НЕ ПРОЙДЕНО")
        
        logger.info("="*60)
    
    async def cleanup(self):
        """Очистка после тестов"""
        await self.client.aclose()


async def main():
    """Главная функция"""
    tester = SystemTester()
    
    try:
        # Подождать немного чтобы система запустилась
        await asyncio.sleep(2)
        
        # Запустить тесты
        results = await tester.run_all_tests()
        
        # Определить код выхода
        total_tests = len(results)
        passed_tests = sum(1 for result in results.values() if result)
        
        if passed_tests == total_tests:
            return 0  # Все тесты пройдены
        elif passed_tests >= total_tests * 0.8:
            return 1  # Большинство тестов пройдено
        else:
            return 2  # Много ошибок
            
    except KeyboardInterrupt:
        logger.info("Тестирование прервано пользователем")
        return 3
    except Exception as e:
        logger.error(f"Критическая ошибка тестирования: {e}")
        return 4
    finally:
        await tester.cleanup()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)