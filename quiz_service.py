import os
import json
import re
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

import PyPDF2
from docx import Document
import requests
from bs4 import BeautifulSoup
import html2text

from ollama_client import OllamaClient

@dataclass
class ContextItem:
    """Элемент контекста"""
    source: str
    content: str
    type: str  # 'text', 'file', 'url'
    id: str = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())[:8]

@dataclass
class QuizConfig:
    """Конфигурация викторины"""
    num_questions: int = 10
    custom_instructions: Optional[str] = None

@dataclass
class QuizQuestion:
    """Вопрос викторины с коротким ответом"""
    id: int
    question: str
    correct_answer: str  # Одно или несколько слов
    explanation: Optional[str] = None
    topic: Optional[str] = None

@dataclass
class QuizResult:
    """Результат генерации викторины"""
    quiz_title: str
    questions: List[QuizQuestion]
    metadata: Dict[str, Any]
    raw_response: Optional[str] = None


class ContentProcessor:
    """Класс для обработки различных типов контента"""
    
    @staticmethod
    def process_text(text: str, max_length: int = 5000) -> str:
        """Обработка простого текста"""
        if len(text) > max_length:
            text = text[:max_length] + f"\n\n[Текст сокращен, всего символов: {len(text)}]"
        return text.strip()
    
    @staticmethod
    def process_file(file_path: str, file_extension: str) -> str:
        """Обработка файлов"""
        if file_extension == '.pdf':
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                return text.strip()
                
        elif file_extension == '.docx':
            doc = Document(file_path)
            return '\n'.join([para.text for para in doc.paragraphs]).strip()
                
        elif file_extension in ['.txt', '.md', '.rtf']:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
                
        else:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            except:
                raise ValueError(f"Неподдерживаемый формат файла: {file_extension}")
    
    @staticmethod
    def process_url(url: str, timeout: int = 10) -> str:
        """
        Обработка HTML страницы по URL
        """
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            
            response = requests.get(
                url, 
                headers=headers, 
                timeout=timeout,
                allow_redirects=True
            )
            response.raise_for_status()
            
            if response.encoding is None:
                response.encoding = 'utf-8'
            
            html_content = response.text
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 
                               'aside', 'form', 'iframe', 'noscript']):
                element.decompose()
            
            title = soup.title.string if soup.title else "Без названия"
            
            text_content = ""
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
            
            if main_content:
                h = html2text.HTML2Text()
                h.ignore_links = False
                h.ignore_images = True
                h.ignore_tables = False
                h.body_width = 0
                
                text_content = h.handle(str(main_content))
            else:
                body = soup.body
                if body:
                    h = html2text.HTML2Text()
                    h.ignore_links = False
                    h.ignore_images = True
                    h.ignore_tables = False
                    h.body_width = 0
                    
                    text_content = h.handle(str(body))
                else:
                    text_content = soup.get_text()
            
            lines = (line.strip() for line in text_content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text_content = '\n'.join(chunk for chunk in chunks if chunk)
            
            result = f"Заголовок: {title}\n\n{text_content}"
            
            max_length = 8000
            if len(result) > max_length:
                result = result[:max_length] + f"\n\n[Текст сокращен, всего символов: {len(result)}]"
            
            return result.strip()
            
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Ошибка загрузки URL {url}: {e}")
        except Exception as e:
            raise ValueError(f"Ошибка обработки HTML: {e}")


class QuizService:
    """
    Сервис для генерации викторин с вопросами, требующими короткого ответа
    """
    
    QUIZ_TEMPLATE = {
        "system_prompt": """Ты - эксперт по созданию образовательных викторин. 
        Твоя задача - создать качественные вопросы для проверки понимания материала.
        
        КРИТИЧЕСКИ ВАЖНЫЕ ИНСТРУКЦИИ:
        1. Используй ТОЛЬКО информацию из предоставленного контекста
        2. Не добавляй информацию из своих общих знаний
        3. Создавай вопросы, требующие краткого ответа (1-3 слова)
        4. Ответ должен быть конкретным и точным
        5. Правильный ответ должен состоять из одного или нескольких слов
        6. Объясни, почему этот ответ правильный
        7. Всегда отвечай на русском языке
        
        Формат вывода должен быть строго в JSON:
        {
            "quiz_title": "Название викторины",
            "questions": [
                {
                    "id": 1,
                    "question": "Вопрос, требующий короткого ответа?",
                    "correct_answer": "Правильный ответ одним или несколькими словами",
                    "explanation": "Пояснение правильного ответа со ссылкой на материал",
                    "topic": "Тема вопроса"
                }
            ]
        }""",
        
        "user_prompt_template": """Создай викторину с вопросами, требующими короткого ответа, на основе следующего материала:
        
        КОНТЕКСТ:
        {context}
        
        ТРЕБОВАНИЯ:
        - Создай {num_questions} вопросов
        - Каждый вопрос должен требовать ответа из 1-3 слов
        - Охвати основные темы материала
        - Включи фактические вопросы и вопросы на понимание
        - Язык: русский
        
        Выведи результат ТОЛЬКО в JSON формате, без дополнительного текста."""
    }
    
    def __init__(self, ollama_client: OllamaClient):
        """
        Инициализация сервиса
        
        Args:
            ollama_client: Клиент Ollama
        """
        self.llm = ollama_client
        self.context: List[ContextItem] = []
        self.content_processor = ContentProcessor()
        
        print(f"[OK] Сервис викторин инициализирован")
        print(f"[INFO] Модель: {ollama_client.get_current_model()}")
        print(f"[INFO] Тип викторины: вопросы с коротким ответом (1-3 слова)")
    
    def add_context_from_file(self, file_path: str) -> str:
        """
        Добавление контекста из файла
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл не найден: {file_path}")
        
        filename = Path(file_path).name
        file_extension = Path(file_path).suffix.lower()
        
        content = self.content_processor.process_file(file_path, file_extension)
        
        context_item = ContextItem(
            source=f"Файл: {filename}",
            content=content,
            type="file"
        )
        
        self.context.append(context_item)
        return context_item.id
    
    def add_context_from_text(self, text: str, source_name: str = "Текст") -> str:
        """
        Добавление контекста из текста
        """
        content = self.content_processor.process_text(text)
        
        context_item = ContextItem(
            source=source_name,
            content=content,
            type="text"
        )
        
        self.context.append(context_item)
        return context_item.id
    
    def add_context_from_url(self, url: str, source_name: Optional[str] = None) -> str:
        """
        Добавление контекста из HTML страницы по URL
        """
        content = self.content_processor.process_url(url)
        
        if source_name is None:
            from urllib.parse import urlparse
            parsed_url = urlparse(url)
            source_name = f"Сайт: {parsed_url.netloc}"
        
        context_item = ContextItem(
            source=source_name,
            content=content,
            type="url"
        )
        
        self.context.append(context_item)
        return context_item.id
    
    def generate_quiz(self, config: QuizConfig) -> QuizResult:
        """
        Генерация викторины на основе контекста
        """
        if not self.context:
            raise ValueError("Нет контекста. Добавьте файлы, текст или URL перед генерацией викторины.")
        
        context_parts = []
        for item in self.context:
            source_type = {
                'text': '[ТЕКСТ]',
                'file': '[ФАЙЛ]',
                'url': '[САЙТ]'
            }.get(item.type, '[ИСТОЧНИК]')
            
            context_parts.append(f"{source_type} ИСТОЧНИК: {item.source}\n{item.content}")
        
        context_text = "\n\n" + "="*50 + "\n\n".join(context_parts) + "\n" + "="*50
        
        user_prompt = self.QUIZ_TEMPLATE["user_prompt_template"].format(
            context=context_text,
            num_questions=config.num_questions
        )
        
        if config.custom_instructions:
            user_prompt += f"\n\nДОПОЛНИТЕЛЬНЫЕ ТРЕБОВАНИЯ:\n{config.custom_instructions}"
        
        messages = [
            {"role": "system", "content": self.QUIZ_TEMPLATE["system_prompt"]},
            {"role": "user", "content": user_prompt}
        ]
        
        print(f"\n[INFO] Генерация викторины...")
        print(f"[INFO] Вопросов: {config.num_questions}")
        print(f"[INFO] Источников: {len(self.context)}")
        print("[INFO] Ожидайте ответа от модели...")
        
        response_text = self.llm.chat(
            messages=messages,
            temperature=0.7,
            max_tokens=4000
        )
        
        if not response_text:
            raise Exception("Модель не вернула ответ")
        
        print("[OK] Викторина сгенерирована!")
        
        quiz_data = self._extract_json_from_response(response_text)
        
        questions = []
        for q_data in quiz_data.get('questions', []):
            questions.append(QuizQuestion(**q_data))
        
        metadata = {
            'num_questions': config.num_questions,
            'model': self.llm.get_current_model(),
            'sources': [
                {
                    'source': item.source,
                    'type': item.type,
                    'id': item.id
                }
                for item in self.context
            ],
            'total_sources': len(self.context),
            'source_types': {
                'text': sum(1 for item in self.context if item.type == 'text'),
                'file': sum(1 for item in self.context if item.type == 'file'),
                'url': sum(1 for item in self.context if item.type == 'url')
            }
        }
        
        result = QuizResult(
            quiz_title=quiz_data.get('quiz_title', 'Викторина с коротким ответом'),
            questions=questions,
            metadata=metadata,
            raw_response=response_text[:1000] if response_text else None
        )
        
        return result
    
    def _extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        """Извлечение JSON из текста ответа"""
        response_text = response_text.replace('```json', '').replace('```', '').strip()
        
        json_pattern = r'\{.*\}'
        matches = re.findall(json_pattern, response_text, re.DOTALL)
        
        if matches:
            try:
                json_str = max(matches, key=len)
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"[WARN] Не удалось распарсить JSON: {e}")
                return self._create_fallback_quiz()
        else:
            print("[WARN] JSON не найден в ответе модели")
            return self._create_fallback_quiz()
    
    def _create_fallback_quiz(self) -> Dict[str, Any]:
        """Создание fallback викторины при ошибке парсинга"""
        return {
            "quiz_title": "Викторина с коротким ответом",
            "questions": [
                {
                    "id": 1,
                    "question": "Пример вопроса, требующего короткого ответа?",
                    "correct_answer": "Пример ответа",
                    "explanation": "Ответ основан на предоставленном материале",
                    "topic": "Основная тема"
                }
            ]
        }
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Получение сводки по контексту"""
        summary = {
            "context_count": len(self.context),
            "total_chars": sum(len(item.content) for item in self.context),
            "sources": []
        }
        
        for item in self.context:
            type_label = {
                'text': 'Текст',
                'file': 'Файл',
                'url': 'Сайт'
            }.get(item.type, 'Источник')
            
            summary["sources"].append({
                "id": item.id,
                "source": f"{type_label}: {item.source}",
                "type": item.type,
                "chars": len(item.content),
                "preview": item.content[:100] + "..." if len(item.content) > 100 else item.content
            })
        
        return summary
    
    def clear_context(self):
        """Очистка контекста"""
        self.context.clear()
        print("[INFO] Контекст очищен")
    
    def remove_context_item(self, item_id: str) -> bool:
        """
        Удаление конкретного элемента контекста
        """
        for i, item in enumerate(self.context):
            if item.id == item_id:
                self.context.pop(i)
                return True
        return False
    
    def export_quiz(self, quiz_result: QuizResult, format: str = "json") -> str:
        """
        Экспорт викторины в различные форматы
        """
        if format == "json":
            return json.dumps(asdict(quiz_result), ensure_ascii=False, indent=2)
        
        elif format == "markdown":
            return self._export_to_markdown(quiz_result)
        
        elif format == "html":
            return self._export_to_html(quiz_result)
        
        else:
            raise ValueError(f"Неподдерживаемый формат: {format}")
    
    def _export_to_markdown(self, quiz_result: QuizResult) -> str:
        """Экспорт в Markdown"""
        md = f"# {quiz_result.quiz_title}\n\n"
        md += "> **Тип викторины:** Вопросы с коротким ответом (1-3 слова)\n\n"
        
        md += "## Метаданные\n\n"
        for key, value in quiz_result.metadata.items():
            if key == 'sources':
                md += f"- **Источники**:\n"
                for source in value:
                    source_type = {
                        'text': 'Текст',
                        'file': 'Файл',
                        'url': 'Веб-страница'
                    }.get(source['type'], 'Источник')
                    md += f"  - {source_type}: {source['source']}\n"
            elif key == 'source_types':
                md += f"- **Типы источников**: "
                types = []
                for type_name, count in value.items():
                    if count > 0:
                        types.append(f"{type_name}: {count}")
                md += ", ".join(types) + "\n"
            else:
                md += f"- **{key}**: {value}\n"
        md += "\n"
        
        md += "## Инструкция\n\n"
        md += "Отвечайте на вопросы одним или несколькими словами. Ответ должен быть точным и соответствовать материалу.\n\n"
        
        md += "## Вопросы\n\n"
        for q in quiz_result.questions:
            md += f"### Вопрос {q.id}\n\n"
            md += f"**{q.question}**\n\n"
            md += f"*Ответ:* `{q.correct_answer}`\n\n"
            
            if q.explanation:
                md += f"**Объяснение:** {q.explanation}\n\n"
            
            if q.topic:
                md += f"**Тема:** {q.topic}\n\n"
            
            md += "---\n\n"
        
        return md
    
    def _export_to_html(self, quiz_result: QuizResult) -> str:
        """Экспорт в HTML"""
        html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{quiz_result.quiz_title}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
            background-color: #f5f7fa;
            color: #333;
        }}
        .quiz-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .quiz-info {{
            background: #e9ecef;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-size: 0.9em;
        }}
        .question {{
            background: white;
            margin: 25px 0;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            border-left: 4px solid #667eea;
        }}
        .question-number {{
            display: inline-block;
            background: #667eea;
            color: white;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            text-align: center;
            line-height: 30px;
            margin-right: 10px;
            font-weight: bold;
        }}
        .answer-box {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            border: 2px dashed #dee2e6;
        }}
        .correct-answer {{
            background: #d4edda;
            border-color: #c3e6cb;
            color: #155724;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            font-weight: bold;
            font-family: 'Courier New', monospace;
            font-size: 1.1em;
        }}
        .explanation {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            border-left: 3px solid #28a745;
        }}
        .metadata {{
            background: #e9ecef;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .source-type {{
            display: inline-block;
            margin-right: 10px;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.8em;
        }}
        .source-text {{ background: #d4edda; color: #155724; }}
        .source-file {{ background: #cce5ff; color: #004085; }}
        .source-url {{ background: #fff3cd; color: #856404; }}
        .instruction {{
            background: #d4edda;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 4px solid #28a745;
        }}
    </style>
</head>
<body>
    <div class="quiz-header">
        <h1>{quiz_result.quiz_title}</h1>
        <p>Викторина с вопросами, требующими короткого ответа (1-3 слова)</p>
    </div>
    
    <div class="instruction">
        <h3>Инструкция:</h3>
        <p>Отвечайте на вопросы одним или несколькими словами. Ответ должен быть точным и соответствовать материалу.</p>
    </div>
"""
        
        html += '<div class="metadata">\n<h3>Метаданные викторины:</h3>\n<ul>\n'
        
        for key, value in quiz_result.metadata.items():
            if key == 'sources':
                html += '<li><strong>Источники:</strong><ul>\n'
                for source in value:
                    source_class = {
                        'text': 'source-text',
                        'file': 'source-file',
                        'url': 'source-url'
                    }.get(source['type'], '')
                    source_label = {
                        'text': 'Текст',
                        'file': 'Файл',
                        'url': 'Веб-страница'
                    }.get(source['type'], 'Источник')
                    
                    html += f'<li><span class="source-type {source_class}">{source_label}</span> {source["source"]}</li>\n'
                html += '</ul></li>\n'
            elif key == 'source_types':
                html += '<li><strong>Типы источников:</strong> '
                types_html = []
                for type_name, count in value.items():
                    if count > 0:
                        types_html.append(f"{type_name}: {count}")
                html += ', '.join(types_html) + '</li>\n'
            else:
                html += f'<li><strong>{key}:</strong> {value}</li>\n'
        
        html += '</ul>\n</div>\n'
        
        html += '<h2>Вопросы</h2>\n'
        
        for q in quiz_result.questions:
            html += f"""
    <div class="question">
        <div>
            <span class="question-number">{q.id}</span>
            <h3 style="display: inline-block;">
                {q.question}
            </h3>
        </div>
        
        <div class="answer-box">
            <p><strong>Место для ответа:</strong> ____________________</p>
        </div>
        
        <div class="correct-answer">
            Правильный ответ: {q.correct_answer}
        </div>
"""
            
            if q.explanation:
                html += f'<div class="explanation"><strong>Объяснение:</strong> {q.explanation}</div>\n'
            
            html += '<div style="margin-top: 20px;">\n'
            if q.topic:
                html += f'<span style="background: #e9ecef; padding: 5px 15px; border-radius: 20px; font-size: 0.8em;">Тема: {q.topic}</span>\n'
            html += '</div>\n'
            
            html += '</div>\n'
        
        html += f"""
    <div style="text-align: center; margin-top: 40px; padding: 20px; background: #f8f9fa; border-radius: 10px;">
        <p>Викторина сгенерирована с помощью Ollama</p>
        <p>Модель: {quiz_result.metadata.get('model', 'Неизвестно')}</p>
    </div>
</body>
</html>"""
        
        return html