from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn
)
from rich.theme import Theme
import logging
import optuna

# Define a custom theme that works well in both light and dark modes
custom_theme = Theme({
    'info': 'dim cyan',
    'warning': 'yellow',
    'error': 'red',
    'success': 'green',
    'progress': 'dim cyan',
    'progress.description': 'dim cyan',
    'status': 'dim cyan',
    'heading': 'dodger_blue2 bold',
    'parameter': 'yellow',
    'value': 'bright_cyan',
})

console = Console(theme=custom_theme)

def setup_rich_logging(log_level=logging.INFO, log_file=None, verbose=False):
    """Configure rich logging with optional file output"""
    # Suppress HTTP request logs from specific libraries
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb.telemetry").setLevel(logging.WARNING)
    logging.getLogger("ragas.testset.evolutions").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("pikepdf._core").setLevel(logging.WARNING)
    
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=console,
                rich_tracebacks=False,
                markup=True,
                show_time=True,
                show_level=True,
                show_path=False
            )
        ]
    )
    
    logger = logging.getLogger("ragbuilder")
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    
    return logger

def get_progress_bar():
    """Create a consistent progress bar style"""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(complete_style='cyan', finished_style='bright_cyan'),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console
    ) 