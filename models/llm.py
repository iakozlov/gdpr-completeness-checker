import os
from typing import List, Optional, Dict, Any
from loguru import logger
from llama_cpp import Llama

from config.llm_config import LlamaConfig, default_config


class LlamaModel:
    """Interface for interacting with Llama models.
    
    Provides methods to generate text responses and symbolic representations
    using llama-cpp-python.
    """
    
    def __init__(self, config: Optional[LlamaConfig] = None):
        """Initialize the Llama model with the given configuration.
        
        Args:
            config: Configuration for the Llama model. If None, uses default config.
            
        Raises:
            FileNotFoundError: If the model file doesn't exist at the specified path.
        """
        self.config = config or default_config
        logger.info(f"Initializing Llama model with config: {self.config}")
        
        if not os.path.exists(self.config.model_path):
            raise FileNotFoundError(f"Model file not found at path: {self.config.model_path}")
        
        self.llm = Llama(
            model_path=self.config.model_path,
            n_gpu_layers=self.config.n_gpu_layers,
            n_threads=4,
            verbose=self.config.verbose,
            n_ctx=self.config.n_ctx
        )
        logger.info("Llama model initialized successfully")
    
    def generate_response(self, prompt: str, max_tokens: int = 2048) -> str:
        """Generate a response from the LLM based on the prompt.
        
        Args:
            prompt: The text prompt to send to the model.
            max_tokens: Maximum number of tokens to generate.
            
        Returns:
            The generated text response.
        """
        logger.debug(f"Generating response for prompt: {prompt[:100]}...")
        
        try:
            response = self.llm.create_chat_completion(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                repeat_penalty=self.config.repeat_penalty,
            )
            
            generated_text = response['choices'][0]['message']['content'].strip()
            logger.debug(f"Generated response: {generated_text[:100]}...")
            return generated_text
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    def generate_symbolic_representation(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a symbolic representation for the given text.
        
        Args:
            user_prompt: The user prompt to send to the model.
            system_prompt: Optional system prompt to specify model behavior.
                           If None, uses a default prompt for deontic logic translation.
            
        Returns:
            The generated symbolic representation.
        """
        if system_prompt is None:
            system_prompt = (
                "You are a specialized AI assistant trained to translate legal requirements "
                "into formal Deontic Logic representations compatible with Deolingo (ASP)."
            )
        
        try:
            response = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2048,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                repeat_penalty=self.config.repeat_penalty,
            )
            
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            logger.error(f"Error generating symbolic representation: {str(e)}")
            raise
    
    def batch_process_texts(self, texts: List[str], system_prompt: Optional[str] = None) -> List[str]:
        """Process multiple texts to generate symbolic representations.
        
        Args:
            texts: List of text prompts to process.
            system_prompt: Optional system prompt to use for all generations.
            
        Returns:
            List of generated symbolic representations.
        """
        results = []
        
        for text in texts:
            try:
                symbolic_repr = self.generate_symbolic_representation(text, system_prompt)
                results.append(symbolic_repr)
            except Exception as e:
                logger.error(f"Error processing text '{text[:50]}...': {str(e)}")
                results.append("")
        
        return results