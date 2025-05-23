import os
import time
from typing import List, Optional, Dict, Any
from loguru import logger
import together
from dotenv import load_dotenv
from together import Together

from config.llama_config import LlamaConfig, default_config

# Load environment variables from .env file
load_dotenv()

class LlamaModel:
    """Interface for interacting with Llama models via Together.ai API.
    
    Provides methods to generate text responses and symbolic representations
    using Together.ai API.
    """
    
    def __init__(self, config: Optional[LlamaConfig] = None):
        """Initialize the Llama model with the given configuration.
        
        Args:
            config: Configuration for the Llama model. If None, uses default config.
            
        Raises:
            ValueError: If the API key is not set or invalid.
        """
        self.config = config or default_config
        logger.info(f"Initializing Llama model with config: {self.config}")
        
        # Set the API key
        api_key = self.config.api_key or os.environ.get("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError("Together API key is not set. Set it in the config, .env file, or as TOGETHER_API_KEY environment variable.")
        
        together.api_key = api_key
        logger.info(f"Llama model initialized successfully with model: {self.config.model}")
        self.client = Together()
    
    def generate_response(self, prompt: str, max_tokens: int = 2048) -> str:
        """Generate a response from the LLM based on the prompt.
        
        Args:
            prompt: The text prompt to send to the model.
            max_tokens: Maximum number of tokens to generate.
            
        Returns:
            The generated text response.
        """
        
        try:
            response = together.Complete.create(
                prompt=prompt,
                model=self.config.model,
                max_tokens=max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
            
            generated_text = response['output']['choices'][0]['text'].strip()
            logger.debug(f"Generated response: {generated_text[:100]}...")
            
            # Implement rate limiting to avoid hitting API limits
            if self.config.rate_limit_delay > 0:
                time.sleep(self.config.rate_limit_delay)
                
            return generated_text
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    def generate_symbolic_representation(self, user_prompt: str, system_prompt: str) -> str:
        """
        Generate symbolic representation using the Together.ai API.
        
        Args:
            user_prompt: The user prompt containing the requirement and DPA segment
            system_prompt: The system prompt with instructions
            
        Returns:
            The generated symbolic representation
        """
        try:
            # Prepare the messages for the chat format
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Make the API call with streaming disabled to get a complete response
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=0.11,
                stream=False  # Explicitly disable streaming
            )
            
            # Extract the response text from a non-streaming response
            result = response.choices[0].message.content.strip()
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating symbolic representation: {str(e)}")
            # Provide more debugging information
            logger.error(f"API parameters: model={self.config.model}, temperature={self.config.temperature}")
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