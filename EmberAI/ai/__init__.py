"""
EmberAI.ai: General AI processing capabilities using Google's Gemini API.

This module provides the core AI processing functionality that powers
the filtering and enrichment features.
"""

import logging
import os
from typing import Optional, Dict, Any
import google.generativeai as genai

logger = logging.getLogger(__name__)

class AIProcessor:
    """Core AI processing class using Google's Gemini API."""
    
    def __init__(self):
        self._model = None
        self._configured = False
        self._api_key = None
        
    def configure(self, GEMINI_API_KEY: str, model: str = "gemini-1.5-flash", **kwargs):
        """
        Configure the AI processor with Gemini API credentials.
        
        Args:
            GEMINI_API_KEY: Your Google AI API key
            model: Gemini model to use (default: gemini-1.5-flash)
            **kwargs: Additional configuration options
        """
        try:
            self._api_key = GEMINI_API_KEY
            genai.configure(api_key=GEMINI_API_KEY)
            
            # Configure generation settings
            generation_config = {
                "temperature": kwargs.get("temperature", 0.1),
                "top_p": kwargs.get("top_p", 0.95),
                "top_k": kwargs.get("top_k", 64),
                "max_output_tokens": kwargs.get("max_output_tokens", 8192),
            }
            
            # Safety settings
            safety_settings = kwargs.get("safety_settings", [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH", 
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
            ])
            
            self._model = genai.GenerativeModel(
                model_name=model,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            self._configured = True
            logger.info(f"✅ EmberAI configured successfully with model: {model}")
            
        except Exception as e:
            logger.error(f"❌ Failed to configure EmberAI: {str(e)}")
            raise RuntimeError(f"EmberAI configuration failed: {str(e)}")
    
    def process(self, prompt: str, **kwargs) -> str:
        """
        Process a prompt using the configured AI model.
        
        Args:
            prompt: The text prompt to process
            **kwargs: Additional generation parameters
            
        Returns:
            str: The AI-generated response
            
        Raises:
            RuntimeError: If not configured or processing fails
        """
        if not self._configured:
            raise RuntimeError(
                "EmberAI not configured. Call EmberAI.ai.configure(GEMINI_API_KEY='your-key') first."
            )
        
        try:
            response = self._model.generate_content(prompt, **kwargs)
            
            if response.candidates and len(response.candidates) > 0:
                return response.text
            else:
                raise RuntimeError("No valid response generated")
                
        except Exception as e:
            logger.error(f"❌ AI processing failed: {str(e)}")
            raise RuntimeError(f"AI processing failed: {str(e)}")
    
    def is_configured(self) -> bool:
        """Check if the AI processor is properly configured."""
        return self._configured
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the configured model."""
        if not self._configured:
            return {"configured": False}
        
        return {
            "configured": True,
            "model": self._model.model_name if self._model else None,
            "api_key_set": bool(self._api_key)
        }

# Global instance
_processor = AIProcessor()

# Module-level functions for easy access
def configure(GEMINI_API_KEY: str, **kwargs):
    """Configure EmberAI with Gemini API credentials."""
    return _processor.configure(GEMINI_API_KEY, **kwargs)

def process(prompt: str, **kwargs) -> str:
    """Process a prompt using the configured AI model."""
    return _processor.process(prompt, **kwargs)

def is_configured() -> bool:
    """Check if EmberAI is properly configured."""
    return _processor.is_configured()

def get_model_info() -> Dict[str, Any]:
    """Get information about the configured model."""
    return _processor.get_model_info()

# Expose the instance for advanced usage
def get_instance() -> AIProcessor:
    """Get the global AIProcessor instance."""
    return _processor

__all__ = [
    "configure",
    "process", 
    "is_configured",
    "get_model_info",
    "get_instance",
    "AIProcessor",
] 