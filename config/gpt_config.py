import os
from dataclasses import dataclass
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class GPTConfig:
    model: str
    api_key: str = None
    temperature: float = 0.1
    top_p: float = 0.9
    rate_limit_delay: float = 0.5  # Sleep time between API calls to avoid rate limits
    
    def __post_init__(self):
        # If api_key is not provided, try to get it from environment variables
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'GPTConfig':
        return cls(**config_dict)

DEFAULT_MODEL = os.environ.get(
    "GPT_MODEL", 
    "gpt-4o-mini"
)

# API key will be loaded from .env file
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY")

default_config = GPTConfig(model=DEFAULT_MODEL, api_key=DEFAULT_API_KEY) 