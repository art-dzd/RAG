"""Демонстрационный скрипт для тестирования RAG API"""

import asyncio
import httpx
import os
from pathlib import Path


async def test_api():
    """Тестирование API"""
    
    base_url = "http://localhost:8000"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        
        print("🚀 Тестирование RAG API")
        print("=" * 50)
        
        # 1. Проверка здоровья системы
        try:
            response = await client.get(f"{base_url}/health")
            if response.status_code == 200:
                print("✅ API сервер доступен")
                print(f"   Статус: {response.json()['status']}")
            else:
                print("❌ API сервер недоступен")
                return
        except Exception as e:
            print(f"❌ Ошибка подключения к API: {e}")
            return
        
        # 2. Создание тестового пользователя
        print("\n📝 Создание тестового пользователя...")
        user_data = {
            "telegram_id": "123456789",
            "username": "testuser",
            "first_name": "Test",
            "last_name": "User"
        }
        
        try:
            response = await client.post(f"{base_url}/users/", json=user_data)
            if response.status_code == 200:
                user_info = response.json()
                print(f"✅ Пользователь создан/найден: {user_info['telegram_id']}")
                print(f"   Новый пользователь: {user_info.get('is_new', False)}")
            else:
                print(f"❌ Ошибка создания пользователя: {response.status_code}")
                return
        except Exception as e:
            print(f"❌ Ошибка при создании пользователя: {e}")
            return
        
        # 3. Получение списка документов
        print("\n📚 Получение списка документов...")
        try:
            response = await client.get(f"{base_url}/users/123456789/documents/")
            if response.status_code == 200:
                docs_info = response.json()
                print(f"✅ Найдено документов: {docs_info['total_documents']}")
                if docs_info['documents']:
                    for doc in docs_info['documents']:
                        print(f"   - {doc['filename']} ({doc['file_type']}, {doc['file_size_mb']:.2f} MB)")
            else:
                print(f"❌ Ошибка получения документов: {response.status_code}")
        except Exception as e:
            print(f"❌ Ошибка при получении документов: {e}")
        
        print("\n✨ Тестирование завершено!")
        print("💡 Для полного тестирования:")
        print("   1. Запустите систему: python run.py")
        print("   2. Откройте Telegram бота")
        print("   3. Отправьте документ и задайте вопросы")


def check_configuration():
    """Проверка конфигурации"""
    print("🔧 Проверка конфигурации")
    print("=" * 50)
    
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ Файл .env не найден")
        print("   Скопируйте .env.example в .env и настройте")
        return False
    
    # Проверить ключевые переменные
    with open(env_file, 'r') as f:
        content = f.read()
    
    checks = [
        ("OPENAI_API_KEY", "your_openai_api_key_here"),
        ("TELEGRAM_BOT_TOKEN", "your_telegram_bot_token_here")
    ]
    
    config_ok = True
    for var_name, default_value in checks:
        if default_value in content:
            print(f"❌ {var_name} не настроен (содержит значение по умолчанию)")
            config_ok = False
        else:
            print(f"✅ {var_name} настроен")
    
    # Проверить структуру проекта
    required_dirs = ["app", "data", "logs"]
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"✅ Директория {dir_name} существует")
        else:
            print(f"❌ Директория {dir_name} не найдена")
            config_ok = False
    
    return config_ok


def main():
    """Главная функция"""
    print("🤖 RAG Telegram Bot - Демонстрация")
    print("=" * 50)
    
    # Проверить конфигурацию
    if not check_configuration():
        print("\n❌ Конфигурация неполная. Настройте систему перед запуском.")
        return
    
    print("\n✅ Конфигурация в порядке!")
    print("\n📋 Следующие шаги:")
    print("1. Запустите систему: python run.py")
    print("2. Дождитесь запуска всех сервисов")
    print("3. Откройте http://localhost:8000/docs для API документации")
    print("4. Найдите вашего бота в Telegram и отправьте /start")
    
    # Спросить, хотят ли протестировать API
    try:
        test_api_input = input("\n🧪 Протестировать API сейчас? (y/N): ").lower()
        if test_api_input in ['y', 'yes', 'да']:
            print("\n⚠️ Убедитесь, что система запущена (python run.py)")
            confirm = input("Система запущена? (y/N): ").lower()
            if confirm in ['y', 'yes', 'да']:
                asyncio.run(test_api())
            else:
                print("💡 Сначала запустите систему, затем выполните: python demo.py")
    except KeyboardInterrupt:
        print("\n👋 До свидания!")


if __name__ == "__main__":
    main() 