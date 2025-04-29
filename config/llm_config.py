# config/llm_config.py
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class LlamaConfig:
    model_path: str
    n_ctx: int = 4096
    n_batch: int = 2048
    n_gpu_layers: int = -1  # -1 means use all available layers
    temperature: float = 0.1
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    verbose: bool = False
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'LlamaConfig':
        return cls(**config_dict)

# Default configuration
DEFAULT_MODEL_PATH = os.environ.get(
    "LLAMA_MODEL_PATH", 
    "models/Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf"  # Default path, adjust as needed
)

default_config = LlamaConfig(
    model_path=DEFAULT_MODEL_PATH,
)