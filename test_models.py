import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pydantic import BaseModel

from ollama_client import OllamaClient

# Pydantic модель для DTO
class QuestionDto(BaseModel):
    """DTO для вопроса"""
    id: int
    questionText: str
    correctAnswer: str

class ModelQuestionAnswer(BaseModel):
    """Ответ модели на вопрос"""
    id: int
    questionText: str
    modelAnswer: str
    correctAnswer: str
    responseTime: float

@dataclass
class ModelTestResult:
    """Результат тестирования модели"""
    modelName: str
    questions: List[ModelQuestionAnswer]
    totalQuestions: int
    totalTime: float
    averageTime: float


class ModelTester:
    """
    Класс для тестирования языковых моделей на вопросах
    """
    
    def __init__(self):
        """Инициализация тестера"""
        self.system_prompt = """Ты - помощник, который отвечает на вопросы.
        Отвечай кратко и точно, используя 1-3 слова.
        Если ты не знаешь ответа, скажи 'Не знаю'.
        Отвечай только на русском языке."""
    
    def test_model_on_questions(self, model_name: str, questions: List[QuestionDto], 
                               base_url: str = "http://localhost:11434") -> ModelTestResult:
        """
        Тестирование модели на списке вопросов
        
        Args:
            model_name: Название модели для тестирования
            questions: Список вопросов для тестирования
            base_url: URL сервера Ollama
            
        Returns:
            Результаты тестирования
        """
        try:
            # Создаем клиент для указанной модели
            client = OllamaClient(base_url=base_url, model=model_name)
            
            results = []
            total_time = 0.0
            
            print(f"[TEST] Тестирование модели: {model_name}")
            print(f"[TEST] Количество вопросов: {len(questions)}")
            
            for question in questions:
                print(f"[TEST] Вопрос {question.id}: {question.questionText[:50]}...")
                
                # Задаем вопрос модели
                start_time = time.time()
                answer = self._ask_question(client, question.questionText)
                response_time = time.time() - start_time
                total_time += response_time
                
                # Сохраняем результат
                result = ModelQuestionAnswer(
                    id=question.id,
                    questionText=question.questionText,
                    modelAnswer=answer,
                    correctAnswer=question.correctAnswer,
                    responseTime=response_time
                )
                results.append(result)
                
                print(f"[TEST]   Ответ модели: {answer}")
                print(f"[TEST]   Время ответа: {response_time:.2f}с")
            
            # Рассчитываем среднее время
            avg_time = total_time / len(questions) if questions else 0
            
            return ModelTestResult(
                modelName=model_name,
                questions=results,
                totalQuestions=len(questions),
                totalTime=total_time,
                averageTime=avg_time
            )
            
        except Exception as e:
            print(f"[ERROR] Ошибка тестирования модели: {e}")
            raise
    
    def _ask_question(self, client: OllamaClient, question: str) -> str:
        """
        Задать вопрос модели
        
        Args:
            client: Клиент Ollama
            question: Текст вопроса
            
        Returns:
            Ответ модели
        """
        try:
            # Подготавливаем промпт
            prompt = f"Вопрос: {question}\n\nОтветь кратко (1-3 слова):"
            
            # Получаем ответ от модели
            response = client.generate(
                prompt=prompt,
                system_prompt=self.system_prompt,
                temperature=0.3,  # Низкая температура для детерминированных ответов
                max_tokens=50
            )
            
            # Очищаем ответ
            response = response.strip()
            
            # Удаляем кавычки и лишние символы
            response = response.replace('"', '').replace("'", '').strip()
            
            # Если ответ слишком длинный, берем первую часть
            if len(response.split()) > 5:
                response = ' '.join(response.split()[:3]) + '...'
            
            return response
            
        except Exception as e:
            print(f"[ERROR] Ошибка при запросе к модели: {e}")
            return "Ошибка при получении ответа"
    
    def batch_test_models(self, model_questions_map: Dict[str, List[QuestionDto]], 
                         base_url: str = "http://localhost:11434") -> Dict[str, ModelTestResult]:
        """
        Пакетное тестирование нескольких моделей
        
        Args:
            model_questions_map: Словарь {имя_модели: список_вопросов}
            base_url: URL сервера Ollama
            
        Returns:
            Словарь с результатами для каждой модели
        """
        results = {}
        
        for model_name, questions in model_questions_map.items():
            try:
                print(f"\n[TEST] Тестирование модели: {model_name}")
                result = self.test_model_on_questions(model_name, questions, base_url)
                results[model_name] = result
                
                # Вывод краткой статистики
                print(f"[TEST] Модель {model_name}:")
                print(f"[TEST]   Всего вопросов: {result.totalQuestions}")
                print(f"[TEST]   Общее время: {result.totalTime:.2f}с")
                print(f"[TEST]   Среднее время на вопрос: {result.averageTime:.2f}с")
                
            except Exception as e:
                print(f"[ERROR] Ошибка тестирования модели {model_name}: {e}")
                results[model_name] = None
        
        return results