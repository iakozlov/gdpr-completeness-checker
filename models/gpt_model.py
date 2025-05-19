import os
import time
from typing import List, Optional, Dict, Any
from loguru import logger
import openai
from dotenv import load_dotenv

from config.gpt_config import GPTConfig, default_config

# Load environment variables from .env file
load_dotenv()

class GPTModel:
    """Interface for interacting with OpenAI GPT models via API.
    
    Provides methods to generate text responses and symbolic representations
    using OpenAI API.
    """
    
    def __init__(self, config: Optional[GPTConfig] = None):
        """Initialize the GPT model with the given configuration.
        
        Args:
            config: Configuration for the GPT model. If None, uses default config.
            
        Raises:
            ValueError: If the API key is not set or invalid.
        """
        self.config = config or default_config
        logger.info(f"Initializing GPT model with config: {self.config}")
        
        # Set the API key
        api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is not set. Set it in the config, .env file, or as OPENAI_API_KEY environment variable.")
        
        openai.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)
        logger.info(f"GPT model initialized successfully with model: {self.config.model}")
    
    def generate_response(self, prompt: str, max_tokens: int = 2048) -> str:
        """Generate a response from the LLM based on the prompt.
        
        Args:
            prompt: The text prompt to send to the model.
            max_tokens: Maximum number of tokens to generate.
            
        Returns:
            The generated text response.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
            
            generated_text = response.choices[0].message.content.strip()
            logger.debug(f"Generated response: {generated_text[:100]}...")
            
            # Implement rate limiting to avoid hitting API limits
            if self.config.rate_limit_delay > 0:
                time.sleep(self.config.rate_limit_delay)
                
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
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2048,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
            
            result = response.choices[0].message.content.strip()
            
            # Implement rate limiting to avoid hitting API limits
            if self.config.rate_limit_delay > 0:
                time.sleep(self.config.rate_limit_delay)
                
            return result
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