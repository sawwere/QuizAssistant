import uvicorn
import os
from api import app

if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "false").lower() == "true"
    
    print("Запуск API для генерации викторин")
    print("=" * 50)
    print(f"Хост: {host}")
    print(f"Порт: {port}")
    print(f"Режим перезагрузки: {reload}")
    print(f"URL Ollama: {os.getenv('OLLAMA_URL', 'http://localhost:11434')}")
    print("=" * 50)
    
    print("\nДокументация API:")
    print(f"Swagger UI: http://{host}:{port}/docs")
    print(f"OpenAPI спецификация: http://{host}:{port}/openapi.json")
    
    print("\nОсобенности:")
    print("  • Автоматический выбор первой доступной модели Ollama")
    print("  • Только один формат: вопросы с коротким ответом (1-3 слова)")
    print("  • Поддержка текста, файлов (PDF, DOCX, TXT) и веб-страниц")
    print("  • Удаление HTML тегов при загрузке URL")
    print("  • Экспорт в JSON, Markdown и HTML форматах")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload
    )