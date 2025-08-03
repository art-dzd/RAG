"""Конфигурация логирования"""

import logging
import sys
from pathlib import Path
from typing import Optional

import structlog


def setup_logging(
    log_level: str = "INFO", 
    log_file: Optional[str] = None,
    enable_json_logs: bool = False
) -> None:
    """
    Настройка системы логирования
    
    Args:
        log_level: Уровень логирования (DEBUG, INFO, WARNING, ERROR)
        log_file: Путь к файлу логов (опционально)
        enable_json_logs: Включить JSON формат логов
    """
    
    # Создать директорию для логов если не существует
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Настройка стандартного логирования
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            *([logging.FileHandler(log_file, encoding='utf-8')] if log_file else [])
        ]
    )
    
    # Настройка structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.dev.ConsoleRenderer() if not enable_json_logs else structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper())
        ),
        logger_factory=structlog.WriteLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Получить логгер с указанным именем
    
    Args:
        name: Имя логгера
        
    Returns:
        Настроенный логгер
    """
    return structlog.get_logger(name)


class PerformanceLogger:
    """Логгер для метрик производительности"""
    
    def __init__(self, logger_name: str = "performance"):
        self.logger = get_logger(logger_name)
        self.metrics = {}
    
    def start_timer(self, operation: str) -> None:
        """Начать отсчет времени для операции"""
        import time
        self.metrics[operation] = {
            'start_time': time.time(),
            'status': 'running'
        }
    
    def end_timer(self, operation: str, success: bool = True, **kwargs) -> float:
        """Завершить отсчет времени и залогировать метрику"""
        import time
        
        if operation not in self.metrics:
            self.logger.warning(f"Timer for operation '{operation}' was not started")
            return 0.0
        
        end_time = time.time()
        duration = end_time - self.metrics[operation]['start_time']
        
        self.logger.info(
            f"Operation completed",
            operation=operation,
            duration_ms=round(duration * 1000, 2),
            success=success,
            **kwargs
        )
        
        del self.metrics[operation]
        return duration
    
    def log_metric(self, metric_name: str, value: float, **kwargs) -> None:
        """Залогировать метрику"""
        self.logger.info(
            f"Metric recorded",
            metric=metric_name,
            value=value,
            **kwargs
        )


# Глобальный экземпляр логгера производительности
performance_logger = PerformanceLogger() 