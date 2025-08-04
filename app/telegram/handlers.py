"""Обработчики команд и сообщений Telegram бота"""

import os
import asyncio
import re
from typing import List, Dict, Any, Optional

from aiogram import Router, F
from aiogram.types import Message, Document, ContentType
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup

import httpx

from app.config import settings
from app.utils.logging_config import get_logger
from app.utils.helpers import sanitize_filename, get_file_size_mb

logger = get_logger(__name__)

# Создать роутер для обработчиков
router = Router()

# API клиент для взаимодействия с FastAPI
API_BASE_URL = f"http://{settings.api_host}:{settings.api_port}"


def escape_markdown(text: str) -> str:
    """
    Экранировать специальные символы для безопасного использования в Markdown
    
    Args:
        text: Текст для экранирования
        
    Returns:
        Экранированный текст
    """
    # Символы, которые нужно экранировать в Markdown
    special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    
    escaped_text = text
    for char in special_chars:
        escaped_text = escaped_text.replace(char, f'\\{char}')
    
    return escaped_text


class UserStates(StatesGroup):
    """Состояния пользователя"""
    waiting_for_document = State()
    chatting_with_document = State()


class APIClient:
    """Клиент для взаимодействия с FastAPI"""
    
    def __init__(self):
        self.base_url = API_BASE_URL
    
    async def create_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Создать или получить пользователя"""
        async with httpx.AsyncClient(timeout=300.0) as client:
            # Добавляем user_id в URL параметры
            user_id = user_data["telegram_id"]
            response = await client.post(f"{self.base_url}/users/?user_id={user_id}", json=user_data)
            response.raise_for_status()
            return response.json()
    
    async def upload_file(self, user_id: str, file_path: str) -> Dict[str, Any]:
        """Загрузить файл"""
        async with httpx.AsyncClient(timeout=300.0) as client:
            with open(file_path, "rb") as f:
                files = {"file": f}
                data = {"user_id": user_id}
                logger.info(f"API запрос: {self.base_url}/upload/ с user_id={user_id}")
                response = await client.post(f"{self.base_url}/upload/", files=files, data=data)
                logger.info(f"API ответ: {response.status_code}")
                response.raise_for_status()
                return response.json()
    
    async def query_document(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """Выполнить запрос к документу"""
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(f"{self.base_url}/query/", json=query_data)
            response.raise_for_status()
            return response.json()
    
    async def get_user_documents(self, user_id: str) -> Dict[str, Any]:
        """Получить список документов пользователя"""
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.get(f"{self.base_url}/users/{user_id}/documents/")
            response.raise_for_status()
            return response.json()
    
    async def delete_document(self, user_id: str, document_id: str) -> Dict[str, Any]:
        """Удалить документ пользователя"""
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.delete(f"{self.base_url}/documents/{document_id}?user_id={user_id}")
            response.raise_for_status()
            return response.json()


# Глобальный API клиент
api_client = APIClient()

# Хранение состояний пользователей
user_contexts: Dict[str, Dict[str, Any]] = {}


def get_user_context(user_id: str) -> Dict[str, Any]:
    """Получить контекст пользователя"""
    if user_id not in user_contexts:
        user_contexts[user_id] = {
            "current_document_id": None,
            "chat_history": [],
            "documents": []
        }
    return user_contexts[user_id]


def add_to_chat_history(user_id: str, role: str, content: str):
    """Добавить сообщение в историю чата"""
    context = get_user_context(user_id)
    context["chat_history"].append({"role": role, "content": content})
    
    # Ограничить историю чата (последние 10 сообщений)
    if len(context["chat_history"]) > 10:
        context["chat_history"] = context["chat_history"][-10:]


@router.message(CommandStart())
async def start_command(message: Message, state: FSMContext):
    """Обработчик команды /start"""
    try:
        user_id = str(message.from_user.id)
        username = message.from_user.username
        first_name = message.from_user.first_name
        last_name = message.from_user.last_name
        
        logger.info(f"Пользователь {user_id} начал диалог")
        
        # Создать или получить пользователя через API
        user_data = {
            "telegram_id": user_id,
            "username": username,
            "first_name": first_name,
            "last_name": last_name
        }
        
        result = await api_client.create_user(user_data)
        is_new = result.get("is_new", False)
        
        # Приветственное сообщение
        welcome_text = f"""
🤖 **Добро пожаловать в RAG-бота!**

{'Рад видеть нового пользователя!' if is_new else 'С возвращением!'}

**Что я умею:**
📄 Обрабатывать документы (PDF, DOCX, TXT)
💬 Отвечать на вопросы по содержимому ваших файлов
🔍 Находить релевантную информацию в больших документах

**Как пользоваться:**
1. Отправьте мне документ (PDF, DOCX или TXT файл)
2. Дождитесь обработки
3. Задавайте вопросы по содержимому!

**Команды:**
/start - Начать работу
/help - Помощь
/documents - Мои документы
/delete - Удалить документ
/clear - Очистить историю чата

**Управление документами:**
📄 /documents - Просмотр всех документов
🗑️ /delete - Удаление документов
🧹 /clear - Очистка истории диалога

Отправьте документ, чтобы начать! 📎
        """
        
        await message.answer(welcome_text, parse_mode="Markdown")
        await state.set_state(UserStates.waiting_for_document)
        
    except Exception as e:
        logger.error(f"Ошибка в start_command: {e}")
        await message.answer(
            "❌ Произошла ошибка при инициализации. Попробуйте позже.",
            parse_mode="Markdown"
        )


@router.message(Command("help"))
async def help_command(message: Message):
    """Обработчик команды /help"""
    help_text = """
🆘 **Справка по использованию RAG-бота**

**Поддерживаемые форматы файлов:**
• PDF документы
• DOCX документы (Word)
• TXT текстовые файлы

**Ограничения:**
• Максимальный размер файла: 50 МБ
• Файлы обрабатываются на сервере безопасно
• Ваши данные изолированы от других пользователей

**Как это работает:**
1. Вы загружаете документ
2. Система анализирует текст и создаёт индекс
3. На ваши вопросы бот отвечает, основываясь на содержимом документа
4. Поддерживается контекст диалога

**Команды:**
/start - Начать работу
/help - Эта справка
/documents - Список ваших документов
/delete - Удалить документ
/clear - Очистить историю текущего чата

**Управление документами:**
📄 /documents - Просмотр всех загруженных документов
🗑️ /delete - Выбор и удаление документа
🧹 /clear - Очистка истории диалога (документы остаются)

**Примеры вопросов:**
• "О чём этот документ?"
• "Какие основные выводы?"
• "Найди информацию о [тема]"
• "Что говорится про [конкретный вопрос]?"

Просто отправьте документ и начинайте задавать вопросы! 🚀
    """
    
    await message.answer(help_text, parse_mode="Markdown")


@router.message(Command("documents"))
async def documents_command(message: Message):
    """Обработчик команды /documents"""
    try:
        user_id = str(message.from_user.id)
        
        # Получить список документов через API
        result = await api_client.get_user_documents(user_id)
        documents = result.get("documents", [])
        
        if not documents:
            await message.answer(
                "📁 У вас пока нет загруженных документов.\n\n"
                "Отправьте документ (PDF, DOCX или TXT), чтобы начать работу!",
                parse_mode="Markdown"
            )
            return
        
        # Сформировать список документов
        docs_text = "📚 **Ваши документы:**\n\n"
        
        for i, doc in enumerate(documents, 1):
            status = "✅ Обработан" if doc["is_processed"] else "⏳ Обрабатывается"
            # Экранировать название файла для безопасного отображения
            safe_filename = escape_markdown(doc['filename'])
            docs_text += (
                f"{i}. **{safe_filename}**\n"
                f"   📊 Тип: {doc['file_type'].upper()}\n"
                f"   📏 Размер: {doc['file_size_mb']:.2f} МБ\n"
                f"   🧩 Частей: {doc['chunks_count']}\n"
                f"   📅 Загружен: {doc['uploaded_at'][:10]}\n"
                f"   {status}\n\n"
            )
        
        docs_text += f"**Всего документов:** {len(documents)}\n\n"
        docs_text += "💡 Отправьте новый документ или задайте вопрос по уже загруженному!"
        
        await message.answer(docs_text, parse_mode="Markdown")
        
    except Exception as e:
        logger.error(f"Ошибка в documents_command: {e}")
        await message.answer(
            "❌ Ошибка при получении списка документов. Попробуйте позже.",
            parse_mode="Markdown"
        )


@router.message(Command("clear"))
async def clear_command(message: Message):
    """Обработчик команды /clear - очищает только историю чата"""
    user_id = str(message.from_user.id)
    
    # Очистить историю чата пользователя
    context = get_user_context(user_id)
    context["chat_history"] = []
    
    await message.answer(
        "🧹 **История чата очищена!**\n\n"
        "История диалога сброшена, но ваши документы остались.\n"
        "Используйте /documents для просмотра документов\n"
        "Используйте /delete для удаления документов",
        parse_mode="Markdown"
    )


@router.message(Command("delete"))
async def delete_command(message: Message):
    """Обработчик команды /delete - показывает список документов для удаления"""
    try:
        user_id = str(message.from_user.id)
        
        # Получить список документов через API
        result = await api_client.get_user_documents(user_id)
        documents = result.get("documents", [])
        
        if not documents:
            await message.answer(
                "📁 У вас нет документов для удаления.\n\n"
                "Сначала загрузите документ (PDF, DOCX или TXT)!",
                parse_mode="Markdown"
            )
            return
        
        # Сформировать список документов с кнопками удаления
        docs_text = "🗑️ **Выберите документ для удаления:**\n\n"
        
        for i, doc in enumerate(documents, 1):
            status = "✅ Обработан" if doc["is_processed"] else "⏳ Обрабатывается"
            # Экранировать название файла для безопасного отображения
            safe_filename = escape_markdown(doc['filename'])
            docs_text += (
                f"{i}. **{safe_filename}**\n"
                f"   📊 Тип: {doc['file_type'].upper()}\n"
                f"   📏 Размер: {doc['file_size_mb']:.2f} МБ\n"
                f"   📅 Загружен: {doc['uploaded_at'][:10]}\n"
                f"   {status}\n\n"
            )
        
        docs_text += (
            f"**Всего документов:** {len(documents)}\n\n"
            "💡 **Для удаления документа отправьте его номер**\n"
            "Например: `1` для удаления первого документа\n\n"
            "⚠️ **Внимание:** Удаление необратимо!"
        )
        
        await message.answer(docs_text, parse_mode="Markdown")
        
    except Exception as e:
        logger.error(f"Ошибка в delete_command: {e}")
        await message.answer(
            "❌ Ошибка при получении списка документов. Попробуйте позже.",
            parse_mode="Markdown"
        )


@router.message(F.text.regexp(r'^\d+$'))
async def handle_document_number(message: Message):
    """Обработчик выбора номера документа для удаления"""
    try:
        user_id = str(message.from_user.id)
        document_number = int(message.text)
        
        # Получить список документов
        result = await api_client.get_user_documents(user_id)
        documents = result.get("documents", [])
        
        if not documents:
            await message.answer(
                "📁 У вас нет документов для удаления.",
                parse_mode="Markdown"
            )
            return
        
        # Проверить корректность номера
        if document_number < 1 or document_number > len(documents):
            await message.answer(
                f"❌ Неверный номер документа.\n\n"
                f"Доступные номера: от 1 до {len(documents)}\n"
                f"Отправьте /delete для просмотра списка документов",
                parse_mode="Markdown"
            )
            return
        
        # Получить выбранный документ
        selected_doc = documents[document_number - 1]
        document_id = selected_doc["id"]
        safe_filename = escape_markdown(selected_doc["filename"])
        
        # Подтверждение удаления
        confirm_text = (
            f"🗑️ **Подтверждение удаления**\n\n"
            f"📄 Документ: **{safe_filename}**\n"
            f"📊 Тип: {selected_doc['file_type'].upper()}\n"
            f"📏 Размер: {selected_doc['file_size_mb']:.2f} МБ\n"
            f"📅 Загружен: {selected_doc['uploaded_at'][:10]}\n\n"
            f"⚠️ **Это действие необратимо!**\n\n"
            f"Для подтверждения отправьте: `да`\n"
            f"Для отмены отправьте: `нет`"
        )
        
        # Сохранить информацию о выбранном документе в контексте
        context = get_user_context(user_id)
        context["pending_delete"] = {
            "document_id": document_id,
            "filename": selected_doc["filename"]
        }
        
        await message.answer(confirm_text, parse_mode="Markdown")
        
    except ValueError:
        await message.answer(
            "❌ Пожалуйста, введите число.\n\n"
            "Отправьте /delete для просмотра списка документов",
            parse_mode="Markdown"
        )
    except Exception as e:
        logger.error(f"Ошибка в handle_document_number: {e}")
        await message.answer(
            "❌ Ошибка при выборе документа. Попробуйте позже.",
            parse_mode="Markdown"
        )


@router.message(F.text.lower().in_(["да", "yes", "удалить", "delete"]))
async def confirm_delete(message: Message):
    """Обработчик подтверждения удаления документа"""
    try:
        user_id = str(message.from_user.id)
        context = get_user_context(user_id)
        
        # Проверить, есть ли ожидающее удаление
        pending_delete = context.get("pending_delete")
        if not pending_delete:
            await message.answer(
                "❌ Нет документа для удаления.\n\n"
                "Отправьте /delete для выбора документа",
                parse_mode="Markdown"
            )
            return
        
        document_id = pending_delete["document_id"]
        filename = pending_delete["filename"]
        safe_filename = escape_markdown(filename)
        
        # Удалить документ через API
        try:
            result = await api_client.delete_document(user_id, document_id)
            
            # Очистить ожидающее удаление
            context.pop("pending_delete", None)
            
            # Если удаляемый документ был текущим, очистить контекст
            if context.get("current_document_id") == document_id:
                context["current_document_id"] = None
                context["chat_history"] = []
            
            await message.answer(
                f"✅ **Документ успешно удалён!**\n\n"
                f"📄 Файл: **{safe_filename}**\n"
                f"🗑️ Удалён из базы данных и векторного хранилища\n\n"
                f"💡 Используйте /documents для просмотра оставшихся документов",
                parse_mode="Markdown"
            )
            
        except Exception as e:
            logger.error(f"Ошибка удаления документа: {e}")
            await message.answer(
                f"❌ Ошибка при удалении документа:\n{str(e)}\n\n"
                f"Попробуйте позже или обратитесь к администратору.",
                parse_mode="Markdown"
            )
            
    except Exception as e:
        logger.error(f"Ошибка в confirm_delete: {e}")
        await message.answer(
            "❌ Произошла ошибка. Попробуйте позже.",
            parse_mode="Markdown"
        )


@router.message(F.text.lower().in_(["нет", "no", "отмена", "cancel"]))
async def cancel_delete(message: Message):
    """Обработчик отмены удаления документа"""
    user_id = str(message.from_user.id)
    context = get_user_context(user_id)
    
    # Очистить ожидающее удаление
    context.pop("pending_delete", None)
    
    await message.answer(
        "❌ **Удаление отменено**\n\n"
        "Документ не был удалён.\n"
        "Используйте /documents для просмотра документов\n"
        "Используйте /delete для удаления документов",
        parse_mode="Markdown"
    )


@router.message(F.document)
async def handle_document(message: Message, state: FSMContext):
    """Обработчик загрузки документов"""
    try:
        user_id = str(message.from_user.id)
        document: Document = message.document
        
        logger.info(f"Получен документ {document.file_name} от пользователя {user_id}")
        
        # Проверить тип файла
        if not document.file_name:
            await message.answer("❌ Не удалось определить имя файла.")
            return
        
        file_extension = document.file_name.split('.')[-1].lower()
        if file_extension not in settings.allowed_file_types:
            await message.answer(
                f"❌ Неподдерживаемый тип файла: {file_extension}\n\n"
                f"Поддерживаемые форматы: {', '.join(settings.allowed_file_types).upper()}",
                parse_mode="Markdown"
            )
            return
        
        # Проверить размер файла
        file_size_mb = document.file_size / (1024 * 1024)
        if file_size_mb > settings.max_file_size_mb:
            await message.answer(
                f"❌ Файл слишком большой: {file_size_mb:.2f} МБ\n\n"
                f"Максимальный размер: {settings.max_file_size_mb} МБ",
                parse_mode="Markdown"
            )
            return
        
        # Экранировать название файла для безопасного отображения
        safe_filename = escape_markdown(document.file_name)
        
        # Отправить уведомление о начале обработки
        processing_message = await message.answer(
            f"⏳ **Обрабатываю документ:** {safe_filename}\n\n"
            f"📊 Размер: {file_size_mb:.2f} МБ\n"
            f"🔄 Это может занять несколько минут...",
            parse_mode="Markdown"
        )
        
        try:
            # Скачать файл
            file_info = await message.bot.get_file(document.file_id)
            file_content = await message.bot.download_file(file_info.file_path)
            
            # Сохранить временно
            temp_dir = f"./data/user_files/{user_id}/temp"
            os.makedirs(temp_dir, exist_ok=True)
            
            safe_filename_for_path = sanitize_filename(document.file_name)
            temp_file_path = os.path.join(temp_dir, safe_filename_for_path)
            
            with open(temp_file_path, "wb") as f:
                f.write(file_content.read())
            
            # Загрузить через API
            logger.info(f"Отправка файла {temp_file_path} для пользователя {user_id}")
            result = await api_client.upload_file(user_id, temp_file_path)
            
            # Удалить временный файл
            os.remove(temp_file_path)
            
            if result["success"]:
                # Обновить контекст пользователя
                context = get_user_context(user_id)
                context["current_document_id"] = result["document_id"]
                context["chat_history"] = []  # Очистить историю для нового документа
                
                # Успешное сообщение с экранированным названием файла
                success_text = (
                    f"✅ **Документ успешно обработан!**\n\n"
                    f"📄 Файл: {safe_filename}\n"
                    f"🧩 Создано частей: {result['chunks_count']}\n"
                    f"⏱ Время обработки: {result['processing_time_seconds']:.1f} сек\n\n"
                    f"💬 Теперь вы можете задавать вопросы по содержимому документа!"
                )
                
                await processing_message.edit_text(success_text, parse_mode="Markdown")
                await state.set_state(UserStates.chatting_with_document)
                
            else:
                await processing_message.edit_text(
                    f"❌ Ошибка при обработке документа:\n{result.get('error', 'Неизвестная ошибка')}",
                    parse_mode="Markdown"
                )
                
        except Exception as e:
            logger.error(f"Ошибка обработки документа: {e}")
            await processing_message.edit_text(
                f"❌ Ошибка при обработке документа:\n{str(e)}",
                parse_mode="Markdown"
            )
            
    except Exception as e:
        logger.error(f"Ошибка в handle_document: {e}")
        await message.answer(
            "❌ Произошла ошибка при загрузке документа. Попробуйте позже.",
            parse_mode="Markdown"
        )


@router.message(F.text)
async def handle_text_message(message: Message, state: FSMContext):
    """Обработчик текстовых сообщений"""
    try:
        user_id = str(message.from_user.id)
        user_text = message.text
        
        context = get_user_context(user_id)
        current_document_id = context.get("current_document_id")
        
        if not current_document_id:
            await message.answer(
                "📄 **Сначала загрузите документ!**\n\n"
                "Отправьте файл (PDF, DOCX или TXT), чтобы я мог отвечать на ваши вопросы.",
                parse_mode="Markdown"
            )
            return
        
        # Показать индикатор набора текста
        await message.bot.send_chat_action(message.chat.id, "typing")
        
        # Добавить вопрос пользователя в историю
        add_to_chat_history(user_id, "user", user_text)
        
        try:
            # Выполнить запрос через API
            query_data = {
                "user_id": user_id,
                "document_id": current_document_id,
                "query": user_text,
                "chat_history": context["chat_history"][:-1]  # Исключить текущий вопрос
            }
            
            result = await api_client.query_document(query_data)
            
            if result["success"]:
                answer = result["answer"]
                found_chunks = result["found_chunks"]
                
                # Добавить ответ в историю
                add_to_chat_history(user_id, "assistant", answer)
                
                # Сформировать ответ
                response_text = f"🤖 **Ответ:**\n\n{answer}\n\n"
                
                if found_chunks > 0:
                    response_text += f"📊 Найдено релевантных частей: {found_chunks}"
                
                await message.answer(response_text, parse_mode="Markdown")
                
            else:
                error_message = result.get("error", "Не найдено релевантной информации")
                await message.answer(
                    f"❌ {error_message}\n\n"
                    "💡 Попробуйте переформулировать вопрос или загрузить другой документ.",
                    parse_mode="Markdown"
                )
                
        except Exception as e:
            logger.error(f"Ошибка выполнения запроса: {e}")
            await message.answer(
                "❌ Ошибка при обработке вашего вопроса. Попробуйте позже.",
                parse_mode="Markdown"
            )
            
    except Exception as e:
        logger.error(f"Ошибка в handle_text_message: {e}")
        await message.answer(
            "❌ Произошла ошибка. Попробуйте позже.",
            parse_mode="Markdown"
        )


# Обработчик для неподдерживаемых типов контента
@router.message()
async def handle_unsupported_content(message: Message):
    """Обработчик неподдерживаемого контента"""
    await message.answer(
        "❌ **Неподдерживаемый тип сообщения**\n\n"
        "Я работаю только с:\n"
        "📄 Документами (PDF, DOCX, TXT)\n"
        "💬 Текстовыми сообщениями\n\n"
        "Отправьте документ или задайте вопрос!",
        parse_mode="Markdown"
    ) 