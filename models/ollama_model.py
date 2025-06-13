import time
from typing import List, Optional, Dict, Any
from loguru import logger

from ollama_client import OllamaClient
from config.ollama_config import OllamaConfig, default_config


class OllamaModel:
    """Interface for interacting with Ollama models via API.
    
    Provides methods to generate text responses and symbolic representations
    using Ollama API.
    """
    
    def __init__(self, config: Optional[OllamaConfig] = None):
        """Initialize the Ollama model with the given configuration.
        
        Args:
            config: Configuration for the Ollama model. If None, uses default config.
            
        Raises:
            ConnectionError: If the Ollama server is not running.
        """
        self.config = config or default_config
        logger.info(f"Initializing Ollama model with config: {self.config}")
        
        # Initialize Ollama client
        self.client = OllamaClient(base_url=self.config.base_url)
        
        # Check if Ollama server is running
        if not self.client.check_health():
            raise ConnectionError(f"Ollama server is not running at {self.config.base_url}")
        
        logger.info(f"Ollama model initialized successfully with model: {self.config.model}")
    
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
            response = self.client.generate(
                prompt=prompt,
                model_name=self.config.model
            )
            
            logger.debug(f"Generated response: {response[:100]}...")
            
            # Implement rate limiting if configured
            if self.config.rate_limit_delay > 0:
                time.sleep(self.config.rate_limit_delay)
                
            return response
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
                "into formal Deontic Logic representations compatible with Deolingo (ASP). "
                "Your task is to convert natural language legal requirements into symbolic "
                "deontic logic expressions using the following format:\n\n"
                "- Use &obligatory{predicate} for obligations\n"
                "- Use &forbidden{predicate} for prohibitions\n"
                "- Use &permitted{predicate} for permissions\n"
                "- Use :- for logical implication\n"
                "- Use role(processor) to indicate the processor role\n"
                "- Create meaningful predicate names that capture the essence of the requirement\n\n"
                "Provide only the symbolic representation without additional explanation."
            )
        
        try:
            response = self.client.generate(
                prompt=user_prompt,
                model_name=self.config.model,
                system_prompt=system_prompt
            )
            
            # Implement rate limiting if configured
            if self.config.rate_limit_delay > 0:
                time.sleep(self.config.rate_limit_delay)
                
            return response
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