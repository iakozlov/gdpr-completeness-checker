# models/llm.py
import os
from typing import List, Dict, Any, Optional
from loguru import logger
from llama_cpp import Llama

from config.llm_config import LlamaConfig, default_config

class LlamaModel:
    def __init__(self, config: Optional[LlamaConfig] = None):
        self.config = config or default_config
        logger.info(f"Initializing Llama model with config: {self.config}")
        
        if not os.path.exists(self.config.model_path):
            raise FileNotFoundError(f"Model file not found at path: {self.config.model_path}")
        
        self.llm = Llama(
            model_path=self.config.model_path,
            n_gpu_layers=-1,
            n_threads = 4,
            verbose=False,
            n_ctx=30000
        )
        logger.info("Llama model initialized successfully")
    
    def generate_response(self, prompt: str, max_tokens: int = 2048) -> str:
        """Generate a response from the LLM based on the prompt."""
        logger.debug(f"Generating response for prompt: {prompt[:100]}...")
        
        # For Instruct models, we'll use the create_chat_completion API
        # which is more appropriate for instruction-tuned models
        response = self.llm.create_chat_completion(
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.1,
            top_p=self.config.top_p,
            repeat_penalty=self.config.repeat_penalty,
        )
        
        generated_text = response['choices'][0]['message']['content'].strip()
        logger.debug(f"Generated response: {generated_text[:100]}...")
        return generated_text
    
    # In models/llm.py, modify the generate_symbolic_representation method

    def generate_symbolic_representation(self, text: str, system_prompt: str = None) -> str:
        """Generate a symbolic representation for the given text."""
        if system_prompt is None:
            system_prompt = (
                "You are a specialized AI assistant trained to translate legal requirements "
                "into formal Deontic Logic representations compatible with Deolingo (ASP)."
            )
        
        # Format specifically for instruct models
        user_instruction = (
            "Translate the following text into a formal Deontic Logic representation using Deolingo syntax. "
            "Use '&obligatory{action}' for obligations (MUST), '&permitted{action}' for permissions (MAY), "
            "and '&forbidden{action}' for prohibitions (MUST NOT). "
            "Format rules with ':-' for implications, ',' for conjunction, ';' for disjunction, and end with periods.\n\n"
            f"Text to translate: {text}\n\n"
            "Return ONLY the logical representation without explanations, examples, or other text. Follow this format:\n"
            "&obligatory{action(actor, object)} :- condition.\n"
            "&forbidden{action(actor, object)} :- not condition."
        )
        
        # Use chat completion API which is more appropriate for instruct models
        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            max_tokens=2048,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            repeat_penalty=self.config.repeat_penalty,
        )
        
        generated_text = response['choices'][0]['message']['content'].strip()
        return generated_text
    
    def batch_process_texts(self, texts: List[str]) -> List[str]:
        """Process multiple texts to generate symbolic representations."""
        results = []
        
        for text in texts:
            symbolic_repr = self.generate_symbolic_representation(text)
            results.append(symbolic_repr)
            
        return results