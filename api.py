from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks, Body
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.openapi.utils import get_openapi
from typing import Optional, List, Dict
import tempfile
import os
from pathlib import Path
from pydantic import BaseModel

from ollama_client import OllamaClient
from quiz_service import QuizService, QuizConfig, QuizResult
from test_models import ModelTester, QuestionDto, ModelTestResult

# Инициализация приложения
app = FastAPI(
    title="API для генерации викторин с коротким ответом",
    description="""API для создания викторин с вопросами, требующими краткого ответа (1-3 слова).
    
    Особенности:
    - Поддержка текста, файлов и веб-страниц
    - Автоматический выбор модели Ollama
    - Викторины с вопросами, требующими короткого ответа
    - Экспорт в JSON, Markdown и HTML форматах
    - Тестирование языковых моделей на пользовательских вопросах
    """,
    contact={
        "name": "Викторина Генератор",
    },
)

# Глобальные объекты
ollama_client = None
quiz_service = None
model_tester = None

# Модели Pydantic для запросов
class ModelTestRequest(BaseModel):
    """Запрос на тестирование модели"""
    model_name: str
    questions: List[QuestionDto]

# Кастомная схема OpenAPI
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version="1.0.0",
        description=app.description,
        routes=app.routes,
    )
    
    openapi_schema["tags"] = [
        {
            "name": "Контекст",
            "description": "Операции с контекстом для викторин",
        },
        {
            "name": "Викторины",
            "description": "Генерация и экспорт викторин",
        },
        {
            "name": "Модели",
            "description": "Управление моделями Ollama",
        },
        {
            "name": "Тестирование",
            "description": "Тестирование моделей на вопросах",
        },
        {
            "name": "Система",
            "description": "Системные эндпоинты",
        }
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске"""
    global ollama_client, quiz_service, model_tester
    
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    
    try:
        # Автоматический выбор модели
        ollama_client = OllamaClient(base_url=ollama_url)
        quiz_service = QuizService(ollama_client)
        model_tester = ModelTester()
        print("[OK] Swagger UI доступен по адресу: http://localhost:8000/docs")
        print("[INFO] OpenAPI спецификация: http://localhost:8000/openapi.json")
        print(f"[INFO] Модель по умолчанию: {ollama_client.get_current_model()}")
    except Exception as e:
        print(f"[ERROR] Ошибка инициализации: {e}")
        raise

@app.get("/", include_in_schema=False)
async def root():
    """Перенаправление на Swagger UI"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta http-equiv="refresh" content="0; url=/docs" />
        <title>API для генерации викторин</title>
    </head>
    <body>
        <p>Перенаправление на <a href="/docs">Swagger UI</a>...</p>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health", tags=["Система"])
async def health_check():
    """
    Проверка здоровья сервиса
    """
    if ollama_client is None or quiz_service is None:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован")
    
    try:
        return {
            "status": "healthy",
            "service": "quiz-generator",
            "model": ollama_client.get_current_model(),
            "context_items": len(quiz_service.context),
            "quiz_type": "short_answer",
            "supported_content_types": ["text", "file", "url"]
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ошибка подключения: {e}")

@app.get("/models", tags=["Модели"])
async def get_available_models():
    """
    Получение списка доступных моделей Ollama
    """
    try:
        models = ollama_client.get_available_models()
        return {
            "available_models": models,
            "current_model": ollama_client.get_current_model()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения списка моделей: {e}")

@app.post("/upload", tags=["Контекст"])
async def upload_file(file: UploadFile = File(...)):
    """
    Загрузка файла для использования в качестве контекста
    """
    try:
        file_extension = Path(file.filename).suffix.lower()
        allowed_extensions = ['.pdf', '.docx', '.txt', '.md', '.rtf']
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Неподдерживаемый формат файла. Разрешены: {', '.join(allowed_extensions)}"
            )
        
        max_size = 10 * 1024 * 1024
        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)
        
        if file_size > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"Файл слишком большой. Максимальный размер: 10MB"
            )
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            context_id = quiz_service.add_context_from_file(tmp_path)
            
            return {
                "success": True,
                "message": "Файл успешно загружен и добавлен в контекст",
                "file_id": context_id,
                "filename": file.filename,
                "file_type": file_extension[1:],
                "file_size": file_size,
                "context_items": len(quiz_service.context)
            }
        finally:
            os.unlink(tmp_path)
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки файла: {str(e)}")

@app.post("/add-text", tags=["Контекст"])
async def add_text(
    text: str = Form(..., description="Текст для добавления в контекст"),
    source_name: str = Form("Текст", description="Название источника")
):
    """
    Добавление текста в контекст
    """
    try:
        if not text.strip():
            raise HTTPException(status_code=400, detail="Текст не может быть пустым")
        
        context_id = quiz_service.add_context_from_text(text, source_name)
        
        return {
            "success": True,
            "message": "Текст успешно добавлен в контекст",
            "context_id": context_id,
            "source_name": source_name,
            "text_length": len(text),
            "context_items": len(quiz_service.context)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка добавления текста: {str(e)}")

@app.post("/add-url", tags=["Контекст"])
async def add_url(
    url: str = Form(..., description="URL веб-страницы"),
    source_name: Optional[str] = Form(None, description="Название источника")
):
    """
    Добавление содержимого веб-страницы по URL в контекст
    """
    try:
        if not url.strip():
            raise HTTPException(status_code=400, detail="URL не может быть пустым")
        
        context_id = quiz_service.add_context_from_url(url, source_name)
        
        return {
            "success": True,
            "message": "Содержимое веб-страницы успешно добавлено в контекст",
            "context_id": context_id,
            "url": url,
            "context_items": len(quiz_service.context)
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка добавления URL: {str(e)}")

@app.get("/context", tags=["Контекст"])
async def get_context():
    """
    Получение информации о текущем контексте
    """
    try:
        summary = quiz_service.get_context_summary()
        return {
            "success": True,
            "context": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения контекста: {str(e)}")

@app.post("/clear", tags=["Контекст"])
async def clear_context():
    """
    Очистка текущего контекста
    """
    try:
        item_count = len(quiz_service.context)
        quiz_service.clear_context()
        return {
            "success": True,
            "message": "Контекст успешно очищен",
            "cleared_items": item_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка очистки контекста: {str(e)}")

@app.delete("/remove/{item_id}", tags=["Контекст"])
async def remove_context_item(item_id: str):
    """
    Удаление конкретного элемента контекста по ID
    """
    try:
        success = quiz_service.remove_context_item(item_id)
        if success:
            return {
                "success": True,
                "message": f"Элемент контекста {item_id} успешно удален",
                "remaining_items": len(quiz_service.context)
            }
        else:
            raise HTTPException(status_code=404, detail=f"Элемент контекста {item_id} не найден")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка удаления элемента контекста: {str(e)}")

@app.post("/generate", tags=["Викторины"])
async def generate_quiz(
    num_questions: int = Form(10, ge=1, le=50, description="Количество вопросов (1-50)"),
    custom_instructions: Optional[str] = Form(None, description="Дополнительные инструкции")
):
    """
    Генерация викторины на основе загруженного контекста
    """
    try:
        if len(quiz_service.context) == 0:
            raise HTTPException(
                status_code=400, 
                detail="Нет загруженного контекста. Сначала загрузите файлы, добавьте текст или URL."
            )
        
        config = QuizConfig(
            num_questions=num_questions,
            custom_instructions=custom_instructions
        )
        
        quiz_result = quiz_service.generate_quiz(config)
        
        result_dict = {
            "quiz_title": quiz_result.quiz_title,
            "questions": [
                {
                    "id": q.id,
                    "question": q.question,
                    "correct_answer": q.correct_answer,
                    "explanation": q.explanation,
                    "topic": q.topic
                }
                for q in quiz_result.questions
            ],
            "metadata": quiz_result.metadata,
            "quiz_type": "short_answer",
            "instructions": "Отвечайте одним или несколькими словами (1-3 слова)"
        }
        
        return {
            "success": True,
            "quiz": result_dict,
            "context_used": len(quiz_service.context),
            "num_questions": num_questions
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации викторины: {str(e)}")

@app.post("/export", tags=["Викторины"])
async def export_quiz(
    quiz_data: dict,
    format: str = Form("json", description="Формат экспорта", enum=["json", "markdown", "html"]),
    background_tasks: BackgroundTasks = None
):
    """
    Экспорт викторины в различные форматы
    """
    try:
        valid_formats = ["json", "markdown", "html"]
        if format not in valid_formats:
            raise HTTPException(
                status_code=400, 
                detail=f"Неподдерживаемый формат. Разрешены: {', '.join(valid_formats)}"
            )
        
        questions = []
        for q_data in quiz_data.get("questions", []):
            class TempQuestion:
                def __init__(self, **kwargs):
                    for key, value in kwargs.items():
                        setattr(self, key, value)
            
            questions.append(TempQuestion(**q_data))
        
        quiz_result = QuizResult(
            quiz_title=quiz_data.get("quiz_title", "Викторина с коротким ответом"),
            questions=questions,
            metadata=quiz_data.get("metadata", {}),
            raw_response=quiz_data.get("raw_response")
        )
        
        exported = quiz_service.export_quiz(quiz_result, format)
        
        if background_tasks:
            file_extension = {
                "json": ".json",
                "markdown": ".md",
                "html": ".html"
            }.get(format, ".txt")
            
            safe_title = "".join(c for c in quiz_result.quiz_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_title = safe_title[:50].replace(' ', '_')
            
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, 
                suffix=file_extension, 
                mode='w', 
                encoding='utf-8'
            )
            temp_file.write(exported)
            temp_file.close()
            
            background_tasks.add_task(os.unlink, temp_file.name)
            
            return FileResponse(
                temp_file.name,
                media_type={
                    "json": "application/json",
                    "markdown": "text/markdown",
                    "html": "text/html"
                }.get(format, "text/plain"),
                filename=f"quiz_{safe_title}{file_extension}"
            )
        else:
            return {
                "success": True,
                "format": format,
                "content": exported[:2000] + ("..." if len(exported) > 2000 else ""),
                "full_length": len(exported)
            }
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка экспорта: {str(e)}")

@app.post("/pull-model", tags=["Модели"])
async def pull_model(
    model_name: str = Form(..., description="Название модели Ollama для загрузки")
):
    """
    Загрузка модели Ollama
    """
    try:
        if not model_name.strip():
            raise HTTPException(status_code=400, detail="Название модели не может быть пустым")
        
        success = ollama_client.pull_model(model_name)
        
        if success:
            return {
                "success": True,
                "message": f"Модель {model_name} успешно загружена"
            }
        else:
            raise HTTPException(status_code=500, detail=f"Не удалось загрузить модель {model_name}")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки модели: {str(e)}")

@app.post("/test/model", tags=["Тестирование"])
async def test_model_on_questions(
    test_request: ModelTestRequest = Body(..., description="Запрос на тестирование модели")
):
    """
    Тестирование указанной модели на списке вопросов
    
    Модель отвечает на каждый вопрос, результаты возвращаются в формате JSON.
    """
    try:
        if model_tester is None:
            raise HTTPException(status_code=503, detail="Модуль тестирования не инициализирован")
        
        # Проверяем входные данные
        if not test_request.model_name.strip():
            raise HTTPException(status_code=400, detail="Название модели не может быть пустым")
        
        if not test_request.questions:
            raise HTTPException(status_code=400, detail="Список вопросов не может быть пустым")
        
        print(f"[TEST] Запуск тестирования модели: {test_request.model_name}")
        print(f"[TEST] Количество вопросов: {len(test_request.questions)}")
        
        # Запускаем тестирование
        test_result = model_tester.test_model_on_questions(
            model_name=test_request.model_name,
            questions=test_request.questions,
            base_url=os.getenv("OLLAMA_URL", "http://localhost:11434")
        )
        
        # Преобразуем результат в словарь для JSON
        result_dict = {
            "modelName": test_result.modelName,
            "questions": [
                {
                    "id": q.id,
                    "questionText": q.questionText,
                    "modelAnswer": q.modelAnswer,
                    "correctAnswer": q.correctAnswer,
                    "responseTime": q.responseTime
                }
                for q in test_result.questions
            ],
            "totalQuestions": test_result.totalQuestions,
            "totalTime": test_result.totalTime,
            "averageTime": test_result.averageTime
        }
        
        return {
            "success": True,
            "testResult": result_dict,
            "message": f"Тестирование завершено. Обработано {test_result.totalQuestions} вопросов."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Ошибка тестирования модели: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка тестирования модели: {str(e)}")

@app.post("/test/models/batch", tags=["Тестирование"])
async def batch_test_models(
    test_requests: Dict[str, List[QuestionDto]] = Body(..., description="Словарь {имя_модели: список_вопросов}")
):
    """
    Пакетное тестирование нескольких моделей
    
    Принимает словарь, где ключ - название модели, значение - список вопросов.
    Возвращает результаты для каждой модели.
    """
    try:
        if model_tester is None:
            raise HTTPException(status_code=503, detail="Модуль тестирования не инициализирован")
        
        if not test_requests:
            raise HTTPException(status_code=400, detail="Словарь тестовых данных не может быть пустым")
        
        print(f"[TEST] Пакетное тестирование {len(test_requests)} моделей")
        
        # Запускаем пакетное тестирование
        results = model_tester.batch_test_models(
            model_questions_map=test_requests,
            base_url=os.getenv("OLLAMA_URL", "http://localhost:11434")
        )
        
        # Преобразуем результаты
        results_dict = {}
        for model_name, result in results.items():
            if result:
                results_dict[model_name] = {
                    "modelName": result.modelName,
                    "totalQuestions": result.totalQuestions,
                    "totalTime": result.totalTime,
                    "averageTime": result.averageTime,
                    "questions": [
                        {
                            "id": q.id,
                            "questionText": q.questionText,
                            "modelAnswer": q.modelAnswer,
                            "correctAnswer": q.correctAnswer,
                            "responseTime": q.responseTime
                        }
                        for q in result.questions
                    ]
                }
            else:
                results_dict[model_name] = {
                    "error": "Не удалось протестировать модель"
                }
        
        return {
            "success": True,
            "results": results_dict,
            "totalModelsTested": len([r for r in results.values() if r]),
            "totalModelsFailed": len([r for r in results.values() if not r])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Ошибка пакетного тестирования: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка пакетного тестирования: {str(e)}")

# Обработчики ошибок
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    import traceback
    traceback.print_exc()
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": f"Внутренняя ошибка сервера: {str(exc)}",
            "status_code": 500
        }
    )