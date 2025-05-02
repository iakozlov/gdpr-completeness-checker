# optimized_llm.py
# This file extends the models/llm.py with optimized settings for the semantic mapping experiment
import os
from typing import List, Dict, Any, Optional
from loguru import logger
from llama_cpp import Llama

from config.llm_config import LlamaConfig, default_config

class OptimizedLlamaModel:
    def __init__(self, config: Optional[LlamaConfig] = None):
        self.config = config or default_config
        logger.info(f"Initializing Optimized Llama model with config: {self.config}")
        
        if not os.path.exists(self.config.model_path):
            raise FileNotFoundError(f"Model file not found at path: {self.config.model_path}")
        
        # Initialize with optimized settings
        self.llm = Llama(
            model_path=self.config.model_path,
            n_gpu_layers=-1,  # Use all available GPU layers
            n_threads=8,  # Increase threads for better performance
            verbose=False,
            n_ctx=8192  # Larger context window for batch processing
        )
        logger.info("Optimized Llama model initialized successfully")
    
    def generate_response(self, prompt: str, max_tokens: int = 4096) -> str:
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
    
    def generate_symbolic_representation(self, text: str, system_prompt: str = None, max_tokens: int = 4096) -> str:
        """
        Generate a symbolic representation with optimized batch processing.
        
        Args:
            text: The text to transform into symbolic representation
            system_prompt: Optional system prompt for the LLM
            max_tokens: Maximum tokens in the response
            
        Returns:
            Generated symbolic representation
        """
        if system_prompt is None:
            system_prompt = (
                "You are a specialized AI assistant trained to translate legal requirements "
                "into formal Deontic Logic representations compatible with Answer Set Programming."
            )
        
        # Increase max_tokens for batch processing
        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            max_tokens=max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            repeat_penalty=self.config.repeat_penalty,
        )
        
        generated_text = response['choices'][0]['message']['content'].strip()
        return generated_text
    
    def batch_process_segments(self, segments: Dict[str, Any], system_prompt: str, prompt_template: str) -> str:
        """
        Process multiple segments in a single LLM call for efficiency.
        
        Args:
            segments: Dictionary of segment data to process
            system_prompt: System prompt for the LLM
            prompt_template: Template for creating the prompt with {segments} placeholder
            
        Returns:
            Combined processing result
        """
        # Format segments data for the prompt
        formatted_segments = ""
        for segment_id, segment_info in segments.items():
            segment_text = segment_info.get("text", "")
            segment_actions = segment_info.get("actions", [])
            formatted_actions = "\n".join(segment_actions)
            
            formatted_segments += f"SEGMENT ID: {segment_id}\n"
            formatted_segments += f"TEXT: {segment_text}\n"
            formatted_segments += f"ACTIONS:\n{formatted_actions}\n\n"
        
        # Create the final prompt
        final_prompt = prompt_template.format(segments=formatted_segments)
        
        # Generate response with increased token limit
        return self.generate_symbolic_representation(
            final_prompt, 
            system_prompt,
            max_tokens=4096  # Increase token limit for batch processing
        )