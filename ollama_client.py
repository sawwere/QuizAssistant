import requests
import json
import time
from typing import List, Dict, Any, Optional

class OllamaClient:
    """
    Клиент для взаимодействия с Ollama API
    """
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = None):
        """
        Инициализация клиента Ollama
        
        Args:
            base_url: URL сервера Ollama
            model: Название модели для использования (если None, будет выбрана первая доступная)
        """
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{self.base_url}/api"
        
        # Проверка подключения и выбор модели
        self.model = model or self._get_first_available_model()
        
        if not self.model:
            raise ConnectionError("Не удалось найти доступные модели Ollama")
        
        print(f"[OK] Используется модель: {self.model}")
    
    def _check_connection(self, retries: int = 5, delay: int = 2) -> bool:
        """Проверка доступности Ollama сервера с повторными попытками"""
        for attempt in range(retries):
            try:
                response = requests.get(f"{self.api_url}/tags", timeout=5)
                if response.status_code == 200:
                    print(f"[OK] Подключено к Ollama: {self.base_url}")
                    return True
                else:
                    print(f"[WARN] Ollama недоступен: {response.status_code}, попытка {attempt + 1}/{retries}")
            except Exception as e:
                print(f"[WARN] Не удалось подключиться к Ollama: {e}, попытка {attempt + 1}/{retries}")
            
            if attempt < retries - 1:
                time.sleep(delay)
        
        return False
    
    def _get_first_available_model(self) -> Optional[str]:
        """Получение первой доступной модели Ollama"""
        try:
            if not self._check_connection():
                raise ConnectionError("Не удалось подключиться к Ollama серверу")
            
            response = requests.get(f"{self.api_url}/tags", timeout=10)
            if response.status_code == 200:
                models_data = response.json()
                models = models_data.get('models', [])
                
                if not models:
                    print("[WARN] На сервере Ollama нет доступных моделей")
                    return None
                
                # Пробуем найти популярные модели в определенном порядке
                preferred_models = [
                    'llama3.2', 'llama3.1', 'llama3',
                    'mistral', 'mixtral',
                    'codellama',
                    'gemma', 'gemma2',
                    'qwen', 'qwen2',
                ]
                
                # Проверяем предпочтительные модели
                for preferred in preferred_models:
                    for model in models:
                        if preferred in model.get('name', '').lower():
                            print(f"[OK] Найдена предпочтительная модель: {model['name']}")
                            return model['name']
                
                # Если предпочтительных нет, берем первую доступную
                first_model = models[0]['name']
                print(f"[OK] Используется первая доступная модель: {first_model}")
                return first_model
            
            return None
            
        except Exception as e:
            print(f"[ERROR] Ошибка при получении списка моделей: {e}")
            return None
    
    def get_available_models(self) -> List[str]:
        """Получение списка всех доступных моделей"""
        try:
            response = requests.get(f"{self.api_url}/tags", timeout=10)
            if response.status_code == 200:
                models_data = response.json()
                return [model['name'] for model in models_data.get('models', [])]
            return []
        except Exception:
            return []
    
    def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.7, 
        max_tokens: int = 4000
    ) -> str:
        """
        Генерация текста через Ollama API
        
        Args:
            prompt: Текст промпта
            system_prompt: Системный промпт
            temperature: Температура генерации (0.0-1.0)
            max_tokens: Максимальное количество токенов
            
        Returns:
            Сгенерированный текст
        """
        try:
            # Формирование полного промпта
            full_prompt = ""
            if system_prompt:
                full_prompt += f"Инструкция системы: {system_prompt}\n\n"
            full_prompt += f"Пользователь: {prompt}\n\nАссистент: "
            
            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            response = requests.post(
                f"{self.api_url}/generate",
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                raise Exception(f"Ошибка Ollama API: {response.status_code} - {response.text}")
                
        except Exception as e:
            raise Exception(f"Ошибка при генерации: {e}")
    
    def chat(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.7, 
        max_tokens: int = 4000
    ) -> str:
        """
        Чат через Ollama API
        
        Args:
            messages: Список сообщений
            temperature: Температура генерации
            max_tokens: Максимальное количество токенов
            
        Returns:
            Ответ модели
        """
        try:
            # Пробуем использовать chat API
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            response = requests.post(
                f"{self.api_url}/chat",
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('message', {}).get('content', '').strip()
            else:
                # Fallback на generate для старых версий Ollama
                return self._chat_fallback(messages, temperature, max_tokens)
                
        except Exception:
            # Fallback на generate при ошибках
            return self._chat_fallback(messages, temperature, max_tokens)
    
    def _chat_fallback(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.7, 
        max_tokens: int = 4000
    ) -> str:
        """Fallback метод для чата через generate API"""
        formatted_prompt = ""
        
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'system':
                formatted_prompt += f"Инструкция системы: {content}\n\n"
            elif role == 'user':
                formatted_prompt += f"Пользователь: {content}\n\n"
            elif role == 'assistant':
                formatted_prompt += f"Ассистент: {content}\n\n"
        
        formatted_prompt += "Ассистент: "
        
        return self.generate(
            prompt=formatted_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    def pull_model(self, model_name: str) -> bool:
        """
        Загрузка модели из репозитория Ollama
        
        Args:
            model_name: Название модели для загрузки
            
        Returns:
            Успешность операции
        """
        try:
            payload = {
                "name": model_name,
                "stream": False
            }
            
            response = requests.post(
                f"{self.api_url}/pull",
                json=payload,
                timeout=300
            )
            
            return response.status_code == 200
        except Exception:
            return False
    
    def get_current_model(self) -> str:
        """Получение текущей используемой модели"""
        return self.model
