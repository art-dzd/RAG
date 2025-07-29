#!/bin/bash

# RAG Telegram Bot - Скрипт запуска
# Убедитесь что вы настроили .env файл перед запуском

echo "🤖 Запуск RAG Telegram Bot системы..."

# Проверить наличие .env файла
if [ ! -f ".env" ]; then
    echo "❌ Файл .env не найден!"
    echo "📝 Скопируйте .env.example в .env и заполните настройки:"
    echo "   cp .env.example .env"
    echo "   nano .env"
    exit 1
fi

# Проверить наличие Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 не найден!"
    echo "📦 Установите Python3:"
    echo "   Ubuntu/Debian: sudo apt install python3 python3-pip python3-venv"
    echo "   CentOS/RHEL: sudo yum install python3 python3-pip"
    exit 1
fi

# Создать виртуальное окружение если не существует
if [ ! -d "venv" ]; then
    echo "📦 Создание виртуального окружения..."
    python3 -m venv venv
    
    if [ $? -ne 0 ]; then
        echo "❌ Не удалось создать виртуальное окружение!"
        echo "📦 Установите python3-venv:"
        echo "   Ubuntu/Debian: sudo apt install python3-venv"
        exit 1
    fi
fi

# Активировать виртуальное окружение
echo "🔄 Активация виртуального окружения..."
source venv/bin/activate

# Обновить pip
pip install --upgrade pip

# Установить зависимости
echo "📚 Установка зависимостей..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Ошибка установки зависимостей!"
    exit 1
fi

# Создать необходимые директории
echo "📁 Создание директорий..."
mkdir -p data/chroma_db
mkdir -p data/user_files
mkdir -p logs

# Проверить конфигурацию
echo "🔍 Проверка конфигурации..."
python3 -c "from app.config import settings; print('✅ Конфигурация в порядке')"

if [ $? -ne 0 ]; then
    echo "❌ Ошибка в конфигурации!"
    echo "📝 Проверьте файл .env"
    exit 1
fi

# Запуск системы
echo "🚀 Запуск системы..."
echo "📡 API будет доступен на: http://localhost:${API_PORT:-8000}"
echo "📋 Логи: ./logs/app.log"
echo "🛑 Для остановки нажмите Ctrl+C"
echo ""

python3 run.py