import os
from dataclasses import dataclass
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class OllamaConfig:
    model: str
    base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    top_p: float = 0.9
    rate_limit_delay: float = 0.1  # Sleep time between API calls
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'OllamaConfig':
        return cls(**config_dict)

DEFAULT_MODEL = os.environ.get(
    "OLLAMA_MODEL", 
    "llama3.3:70b"
)

DEFAULT_BASE_URL = os.environ.get(
    "OLLAMA_BASE_URL",
    "http://localhost:11434"
)

default_config = OllamaConfig(
    model=DEFAULT_MODEL, 
    base_url=DEFAULT_BASE_URL
) 