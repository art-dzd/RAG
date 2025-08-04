# RAG Telegram Bot

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.6-green.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.12-orange.svg)](https://langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**RAG Telegram Bot — chat with your documents**

Telegram bot with Retrieval-Augmented Generation (RAG) for document conversation (PDF, DOCX, TXT) in natural language. Upload files and ask questions — the bot responds with context from your documents using vector search and GPT-4.1-mini.

**🧠 What's under the hood**

Modern stack for AI interfaces in Python:

**FastAPI** — API server with async architecture

**LangChain** — RAG logic and processing chains

**ChromaDB** — vector storage for semantic search

**OpenAI GPT-4.1-mini** — context-aware response generation

**aiogram 3.x** — Telegram interface

**SQLite** — database for metadata and history storage

**🧪 Usage examples**

**📚 Education**
- Ask questions about uploaded textbooks
- Prepare for exams based on notes
- Convert PDF courses to dialog format

**💼 Document work**
- Quick information search in manuals, reports
- Answers on uploaded regulations and policies
- Generate summaries of long texts

**👨‍💻 Product/AI development**
- RAG implementation example with production architecture
- Ready integration Telegram + FastAPI + OpenAI
- Suitable as foundation for SaaS or internal assistant

## 🎯 **Use Cases**

### **📚 Document Q&A**
- Upload research papers, manuals, reports
- Ask contextual questions about content
- Get AI-powered summaries and insights

### **💼 Business Intelligence**
- Process company documents and policies
- Extract key information and trends
- Generate reports and summaries

### **🎓 Educational Content**
- Upload textbooks and course materials
- Create interactive learning experiences
- Provide instant answers to student questions

## 🚀 **Key Features**

### **🏗️ Modern Architecture**
- **Async-first design** with FastAPI + aiogram 3.x
- **Microservices pattern** with isolated service layers
- **Production-ready** with comprehensive error handling
- **Scalable** multi-user system with data isolation

### **🤖 AI/ML Excellence**
- **RAG implementation** with LangChain 0.3.x
- **Vector search** using ChromaDB for semantic similarity
- **GPT-4.1-mini integration** with streaming responses
- **Smart document chunking** with configurable parameters

### **🔒 Enterprise Security**
- **Input validation** with Pydantic 2.x
- **Rate limiting** and CORS protection
- **Path traversal prevention** and XSS protection
- **Secure file handling** with MIME validation

### **📊 Production Monitoring**
- **Health checks** and metrics endpoints
- **Structured logging** with correlation IDs
- **Performance monitoring** and error tracking
- **Comprehensive API documentation**

## 🏗️ **Architecture Overview**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Telegram Bot  │    │   FastAPI API   │    │   External      │
│   (aiogram 3.x) │    │   (Security +   │    │   Clients       │
│                 │    │   Rate Limiting)│    │                 │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │      RAG Service Layer    │
                    │  (Document Processing +   │
                    │   Vector Search + AI)     │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │      Data Layer           │
                    │  (SQLite + ChromaDB +    │
                    │   File Storage)           │
                    └───────────────────────────┘
```

## 🛠️ **Tech Stack**

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **Backend** | FastAPI | 0.115.6 | High-performance async API |
| **Bot Framework** | aiogram | 3.15.0 | Modern Telegram bot |
| **AI/ML** | LangChain + OpenAI | 0.3.12 + 1.54.4 | RAG orchestration |
| **Vector DB** | ChromaDB | 0.5.23 | Semantic search |
| **Database** | SQLite + SQLAlchemy | 2.0.36 | Data persistence |
| **Validation** | Pydantic | 2.9.2 | Data validation |
| **Security** | slowapi + bleach | Latest | Rate limiting + sanitization |

## ⚡ **Quick Start**

### **1. Setup Environment**
```bash
git clone <repository-url>
cd RAG
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### **2. Configure**
```bash
cp .env.example .env
# Edit .env with your API keys:
# OPENAI_API_KEY=sk-your-key
# TELEGRAM_BOT_TOKEN=your-bot-token
```

### **3. Run**
```bash
python run.py
```

**Access:**
- 🌐 **API**: http://localhost:8000
- 📚 **Docs**: http://localhost:8000/docs
- 🤖 **Telegram**: @your_bot_name

## 📡 **API Endpoints**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System health check |
| `/metrics` | GET | Usage statistics |
| `/upload/` | POST | Document processing |
| `/query/` | POST | RAG-powered Q&A |
| `/users/{id}/documents/` | GET | User documents |

## 🔧 **Configuration**

**Core Settings:**
```env
# Required
OPENAI_API_KEY=sk-your-key
TELEGRAM_BOT_TOKEN=your-token

# RAG Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=5

# Security
RATE_LIMIT_ENABLED=true
MAX_REQUESTS_PER_MINUTE=30

# Performance
MAX_CONCURRENT_REQUESTS=10
REQUEST_TIMEOUT=300
```

## 📊 **Performance & Scalability**

### **✅ Optimized For:**
- **Concurrent users**: 100+ simultaneous users
- **Document size**: Up to 50MB per file
- **Response time**: <2s for typical queries
- **Memory usage**: Efficient async processing
- **Storage**: Local SQLite + ChromaDB

### **📈 Scalability Features:**
- **Async architecture** for high concurrency
- **User isolation** with separate vector collections
- **Configurable limits** for resource management
- **Graceful degradation** under load

## 🔒 **Security Features**

- **Input validation** with Pydantic schemas
- **Rate limiting** per user/IP
- **File type validation** and size limits
- **Path traversal protection**
- **XSS prevention** with text sanitization
- **CORS middleware** for web clients
- **TrustedHost middleware** for production

## 📈 **Monitoring & Observability**

- **Health checks** for all services
- **Structured logging** with correlation IDs
- **Performance metrics** and response times
- **Error tracking** with detailed context
- **Usage statistics** and user analytics

## 🚀 **Deployment Ready**

### **Production Features:**
- ✅ **Systemd service** configuration
- ✅ **Docker support** with multi-stage builds
- ✅ **Environment-based** configuration
- ✅ **Graceful shutdown** handling
- ✅ **Log rotation** and management
- ✅ **Health monitoring** endpoints

### **Deployment Options:**
```bash
# Systemd (Linux)
sudo systemctl enable rag-bot
sudo systemctl start rag-bot

# Docker
docker build -t rag-bot .
docker run -p 8000:8000 rag-bot

# Direct
python run.py
```

## 🧪 **Testing**

```bash
# System health check
curl http://localhost:8000/health

# API documentation
open http://localhost:8000/docs

# Telegram bot test
# Send /start to your bot
```

## 📁 **Project Structure**

```
RAG/
├── app/
│   ├── config.py              # Configuration management
│   ├── main.py               # FastAPI application
│   ├── database/             # Database models & connection
│   ├── services/             # Business logic
│   │   ├── rag_service.py    # RAG orchestration
│   │   ├── openai_service.py # OpenAI API integration
│   │   ├── vector_store.py   # ChromaDB operations
│   │   └── file_parser.py    # Document parsing
│   ├── telegram/             # Telegram bot
│   └── utils/               # Utilities & helpers
├── data/                    # Data storage
├── logs/                    # Application logs
└── requirements.txt         # Dependencies
```

## 📄 **License**

MIT License - see [LICENSE](LICENSE) for details.

---

## 🇷🇺 **RAG Telegram Bot**

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.6-green.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.12-orange.svg)](https://langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**RAG Telegram Bot — чат с вашими документами**

Телеграм-бот с Retrieval-Augmented Generation (RAG) для общения с документами (PDF, DOCX, TXT) на естественном языке. Загружайте файлы и задавайте вопросы — бот отвечает с учётом их содержимого, используя векторный поиск и GPT-4.1-mini.

**🧠 Что под капотом**

Современный стек для AI-интерфейсов на Python:

**FastAPI** — API сервер с асинхронной архитектурой

**LangChain** — RAG-логика и цепочки обработки

**ChromaDB** — векторное хранилище для семантического поиска

**OpenAI GPT-4.1-mini** — генерация ответов с учётом контекста

**aiogram 3.x** — Telegram-интерфейс

**SQLite** — база для хранения метаданных и истории

**🧪 Примеры использования**

**📚 Образование**
- Задавать вопросы по загруженным учебникам
- Готовиться к экзаменам на основе конспектов
- Переводить PDF-курсы в диалоговый формат

**💼 Работа с документами**
- Быстрый поиск информации в инструкциях, отчётах
- Ответы по загруженным регламентам и политикам
- Генерация резюме по длинным текстам

**👨‍💻 Продукт/AI-разработка**
- Пример RAG-реализации с продакшн-архитектурой
- Готовая интеграция Telegram + FastAPI + OpenAI
- Подходит как основа для SaaS или внутреннего ассистента

## 🎯 **Как можно использовать**

### **📚 Документальные Q&A**
- Загружайте исследовательские работы, мануалы, отчеты
- Задавайте контекстные вопросы о содержании
- Получайте AI-powered резюме и инсайты

### **💼 Бизнес-аналитика**
- Обрабатывайте корпоративные документы и политики
- Извлекайте ключевую информацию и тренды
- Генерируйте отчеты и резюме

### **🎓 Образовательный контент**
- Загружайте учебники и материалы курсов
- Создавайте интерактивные обучающие опыты
- Предоставляйте мгновенные ответы на вопросы студентов

## 🚀 **Ключевые возможности**

### **🏗️ Современная архитектура**
- **Асинхронный дизайн** с FastAPI + aiogram 3.x
- **Микросервисная архитектура** с изолированными слоями
- **Готовность к продакшену** с комплексной обработкой ошибок
- **Масштабируемость** для множественных пользователей

### **🤖 AI/ML превосходство**
- **RAG реализация** с LangChain 0.3.x
- **Векторный поиск** используя ChromaDB для семантического поиска
- **Интеграция GPT-4.1-mini** с потоковыми ответами
- **Умное разбиение документов** с настраиваемыми параметрами

### **🔒 Корпоративная безопасность**
- **Валидация входных данных** с Pydantic 2.x
- **Ограничение запросов** и CORS защита
- **Предотвращение обхода путей** и XSS защита
- **Безопасная обработка файлов** с MIME валидацией

### **📊 Мониторинг продакшена**
- **Проверки здоровья** и метрики
- **Структурированное логирование** с correlation ID
- **Мониторинг производительности** и отслеживание ошибок
- **Комплексная документация API**

## 🛠️ **Технологический стек**

| Компонент | Технология | Версия | Назначение |
|-----------|------------|---------|---------|
| **Backend** | FastAPI | 0.115.6 | Высокопроизводительный async API |
| **Bot Framework** | aiogram | 3.15.0 | Современный Telegram бот |
| **AI/ML** | LangChain + OpenAI | 0.3.12 + 1.54.4 | RAG оркестрация |
| **Vector DB** | ChromaDB | 0.5.23 | Семантический поиск |
| **Database** | SQLite + SQLAlchemy | 2.0.36 | Персистентность данных |
| **Validation** | Pydantic | 2.9.2 | Валидация данных |
| **Security** | slowapi + bleach | Latest | Rate limiting + санитизация |

## ⚡ **Быстрый старт**

### **1. Настройка окружения**
```bash
git clone <repository-url>
cd RAG
python -m venv venv
source venv/bin/activate  # Linux/macOS
# или venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### **2. Конфигурация**
```bash
cp .env.example .env
# Отредактируйте .env с вашими API ключами:
# OPENAI_API_KEY=sk-your-key
# TELEGRAM_BOT_TOKEN=your-bot-token
```

### **3. Запуск**
```bash
python run.py
```

**Доступ:**
- 🌐 **API**: http://localhost:8000
- 📚 **Документация**: http://localhost:8000/docs
- 🤖 **Telegram**: @your_bot_name

## 📡 **API эндпоинты**

| Эндпоинт | Метод | Описание |
|----------|--------|-------------|
| `/health` | GET | Проверка состояния системы |
| `/metrics` | GET | Статистика использования |
| `/upload/` | POST | Обработка документов |
| `/query/` | POST | RAG-powered вопросы и ответы |
| `/users/{id}/documents/` | GET | Документы пользователя |

## 🔧 **Конфигурация**

**Основные настройки:**
```env
# Обязательные
OPENAI_API_KEY=sk-your-key
TELEGRAM_BOT_TOKEN=your-token

# RAG конфигурация
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=5

# Безопасность
RATE_LIMIT_ENABLED=true
MAX_REQUESTS_PER_MINUTE=30

# Производительность
MAX_CONCURRENT_REQUESTS=10
REQUEST_TIMEOUT=300
```

## 📊 **Производительность и масштабируемость**

### **✅ Оптимизировано для:**
- **Одновременных пользователей**: 100+ пользователей
- **Размера документов**: До 50MB на файл
- **Времени ответа**: <2с для типичных запросов
- **Использования памяти**: Эффективная async обработка
- **Хранилища**: Локальная SQLite + ChromaDB

### **📈 Особенности масштабируемости:**
- **Асинхронная архитектура** для высокой конкурентности
- **Изоляция пользователей** с отдельными векторными коллекциями
- **Настраиваемые лимиты** для управления ресурсами
- **Graceful degradation** под нагрузкой

## 🔒 **Функции безопасности**

- **Валидация входных данных** с Pydantic схемами
- **Ограничение запросов** на пользователя/IP
- **Валидация типов файлов** и лимиты размера
- **Защита от обхода путей**
- **Предотвращение XSS** с санитизацией текста
- **CORS middleware** для веб-клиентов
- **TrustedHost middleware** для продакшена

## 📈 **Мониторинг и наблюдаемость**

- **Проверки здоровья** для всех сервисов
- **Структурированное логирование** с correlation ID
- **Метрики производительности** и время ответа
- **Отслеживание ошибок** с детальным контекстом
- **Статистика использования** и аналитика пользователей

## 🚀 **Готовность к развертыванию**

### **Продакшен функции:**
- ✅ **Systemd сервис** конфигурация
- ✅ **Docker поддержка** с multi-stage builds
- ✅ **Конфигурация на основе окружения**
- ✅ **Graceful shutdown** обработка
- ✅ **Ротация логов** и управление
- ✅ **Health monitoring** эндпоинты

### **Варианты развертывания:**
```bash
# Systemd (Linux)
sudo systemctl enable rag-bot
sudo systemctl start rag-bot

# Docker
docker build -t rag-bot .
docker run -p 8000:8000 rag-bot

# Прямой запуск
python run.py
```

## 🧪 **Тестирование**

```bash
# Проверка здоровья системы
curl http://localhost:8000/health

# Документация API
open http://localhost:8000/docs

# Тест Telegram бота
# Отправьте /start вашему боту
```

## 📁 **Структура проекта**

```
RAG/
├── app/
│   ├── config.py              # Управление конфигурацией
│   ├── main.py               # FastAPI приложение
│   ├── database/             # SQLAlchemy модели
│   ├── services/             # Бизнес-логика
│   │   ├── rag_service.py    # RAG оркестрация
│   │   ├── openai_service.py # AI интеграция
│   │   ├── vector_store.py   # ChromaDB операции
│   │   └── file_parser.py    # Обработка документов
│   ├── telegram/             # Обработчики бота
│   └── utils/               # Общие утилиты
├── data/                    # Постоянное хранилище
├── logs/                    # Логи приложения
└── requirements.txt         # Зависимости
```

## 📄 **Лицензия**

MIT License - см. [LICENSE](LICENSE) для деталей. 