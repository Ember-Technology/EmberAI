"""
EmberAI: Comprehensive AI processing system with filtering and enrichment capabilities.

Usage:
    # Configure
    import EmberAI
    EmberAI.ai.configure(GEMINI_API_KEY="your-api-key")
    EmberAI.filters.configure()  # Optional filter configurations
    
    # General AI processing
    response = EmberAI.ai.process("Your prompt here")
    
    # AI-powered filtering
    result = await EmberAI.filters.process(users, block_eu_users=True, enrich_gender=True)
"""

from . import ai
from . import filters

__version__ = "1.0.0"
__author__ = "Saqib Khan Afridi"

__all__ = [
    "ai",
    "filters",
] 