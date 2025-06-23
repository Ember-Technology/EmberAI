"""
EmberAI.filters: AI-powered filtering and enrichment system.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import logging
import time
import asyncio
from aiolimiter import AsyncLimiter
from .ai import get_instance as get_ai_instance

logger = logging.getLogger(__name__)

@dataclass
class FilterConfig:
    """Configuration for AI filters with toggles."""
    block_eu_users: bool = False
    block_legal_users: bool = False
    block_unique_names: bool = False
    enrich_gender: bool = False
    preserve_order: bool = True
    log_stats: bool = True
    batch_size: int = 25
    rate_limit_per_sec: int = 50
    max_concurrent: int = 128

class FilterProcessor:
    """AI-powered filtering and enrichment processor."""
    
    EU_NAMES = [
        "Müller", "Schmidt", "Schneider", "García", "González", "Martin", "Bernard",
        "Rossi", "Russo", "Ferrari", "Nielsen", "Hansen", "Andersson", "Larsson"
    ]
    
    def __init__(self, config: FilterConfig):
        self.config = config
        self._limiter = AsyncLimiter(config.rate_limit_per_sec, time_period=1)
        self._semaphore = asyncio.Semaphore(config.max_concurrent)
    
    async def process_filters(self, users: List[Dict]) -> Dict:
        """Process users with AI-powered filtering and enrichment."""
        ai_processor = get_ai_instance()
        if not ai_processor.is_configured():
            raise RuntimeError("EmberAI not configured. Call EmberAI.ai.configure() first.")
        
        if not users:
            return {
                'processed_users': [],
                'original_count': 0,
                'final_count': 0,
                'processing_stats': {},
                'filters_applied': []
            }
        
        start_time = time.time()
        current_users = users.copy() if self.config.preserve_order else users
        filters_applied = []
        
        # Apply filters based on configuration
        if self.config.block_eu_users:
            current_users = await self._filter_eu_users(current_users)
            filters_applied.append("block_eu_users")
        
        if self.config.block_legal_users:
            current_users = await self._filter_legal_users(current_users)
            filters_applied.append("block_legal_users")
        
        if self.config.enrich_gender:
            current_users = await self._enrich_gender(current_users)
            filters_applied.append("enrich_gender")
        
        total_time = time.time() - start_time
        
        return {
            'processed_users': current_users,
            'original_count': len(users),
            'final_count': len(current_users),
            'filters_applied': filters_applied,
            'processing_stats': {
                'total_time_seconds': total_time,
                'processing_rate': len(users) / total_time if total_time > 0 else 0
            }
        }
    
    async def _filter_eu_users(self, users: List[Dict]) -> List[Dict]:
        """Filter out EU users using AI classification."""
        if not users:
            return users
        
        # Build prompt for EU classification
        prompt = self._build_eu_prompt(users)
        
        # Process with AI
        ai_processor = get_ai_instance()
        async with self._semaphore, self._limiter:
            response = await asyncio.to_thread(ai_processor.process, prompt)
        
        # Parse response and filter
        classifications = self._parse_eu_response(response, len(users))
        return [u for u, is_eu in zip(users, classifications) if not is_eu]
    
    async def _filter_legal_users(self, users: List[Dict]) -> List[Dict]:
        """Filter out legal professionals using AI classification."""
        if not users:
            return users
        
        prompt = self._build_legal_prompt(users)
        
        ai_processor = get_ai_instance()
        async with self._semaphore, self._limiter:
            response = await asyncio.to_thread(ai_processor.process, prompt)
        
        classifications = self._parse_legal_response(response, len(users))
        return [u for u, is_legal in zip(users, classifications) if not is_legal]
    
    async def _enrich_gender(self, users: List[Dict]) -> List[Dict]:
        """Enrich users with gender information using AI classification."""
        if not users:
            return users
        
        prompt = self._build_gender_prompt(users)
        
        ai_processor = get_ai_instance()
        async with self._semaphore, self._limiter:
            response = await asyncio.to_thread(ai_processor.process, prompt)
        
        genders = self._parse_gender_response(response, len(users))
        
        # Add gender to users
        enriched_users = []
        for user, gender in zip(users, genders):
            enriched_user = dict(user)
            enriched_user['gender'] = gender
            enriched_users.append(enriched_user)
        
        return enriched_users
    
    def _build_eu_prompt(self, users: List[Dict]) -> str:
        """Build prompt for EU classification."""
        system_prompt = """Classify if users are EU residents. Return exactly one per line:
<classification>EU</classification> or <classification>NOT_EU</classification>"""
        
        input_blocks = []
        for idx, user in enumerate(users, start=1):
            input_blocks.append(f"""<input {idx}>
Location: {user.get('location', '')}
Name: {user.get('name', '')}
Description: {user.get('description', '')}
</input>""")
        
        return f"{system_prompt}\n\n" + "\n".join(input_blocks)
    
    def _build_legal_prompt(self, users: List[Dict]) -> str:
        """Build prompt for legal profession classification."""
        system_prompt = """Classify if users are legal professionals. Return exactly one per line:
<classification>LEGAL</classification> or <classification>NOT_LEGAL</classification>"""
        
        input_blocks = []
        for idx, user in enumerate(users, start=1):
            input_blocks.append(f"""<input {idx}>
Username: {user.get('username', '')}
Name: {user.get('full_name', '')}
Description: {user.get('description', '')}
Education: {user.get('education', '')}
Employer: {user.get('employer', '')}
</input>""")
        
        return f"{system_prompt}\n\n" + "\n".join(input_blocks)
    
    def _build_gender_prompt(self, users: List[Dict]) -> str:
        """Build prompt for gender classification."""
        system_prompt = """Classify gender from names/usernames. Return exactly one per line:
<gender>male</gender>, <gender>female</gender>, or <gender>unknown</gender>"""
        
        input_blocks = []
        for idx, user in enumerate(users, start=1):
            name = user.get('full_name') or user.get('name', '')
            username = user.get('username', '')
            input_blocks.append(f"""<input {idx}>
Name: {name}
Username: {username}
</input>""")
        
        return f"{system_prompt}\n\n" + "\n".join(input_blocks)
    
    def _parse_eu_response(self, response: str, expected: int) -> List[bool]:
        """Parse EU classification response."""
        lines = [line.strip() for line in response.splitlines() if line.strip()]
        classifications = []
        
        for line in lines:
            if line == "<classification>EU</classification>":
                classifications.append(True)
            elif line == "<classification>NOT_EU</classification>":
                classifications.append(False)
        
        return classifications[:expected]
    
    def _parse_legal_response(self, response: str, expected: int) -> List[bool]:
        """Parse legal classification response."""
        lines = [line.strip() for line in response.splitlines() if line.strip()]
        classifications = []
        
        for line in lines:
            if line == "<classification>LEGAL</classification>":
                classifications.append(True)
            elif line == "<classification>NOT_LEGAL</classification>":
                classifications.append(False)
        
        return classifications[:expected]
    
    def _parse_gender_response(self, response: str, expected: int) -> List[str]:
        """Parse gender classification response."""
        lines = [line.strip() for line in response.splitlines() if line.strip()]
        genders = []
        
        for line in lines:
            if line.startswith("<gender>") and line.endswith("</gender>"):
                gender = line[8:-9].strip().lower()
                genders.append(gender if gender in ["male", "female", "unknown"] else "unknown")
        
        return genders[:expected]

# Global configuration
_config = FilterConfig()
_processor = None

def configure(**kwargs):
    """Configure the filters module."""
    global _config, _processor
    for key, value in kwargs.items():
        if hasattr(_config, key):
            setattr(_config, key, value)
    _processor = FilterProcessor(_config)
    logger.info("✅ EmberAI Filters configured")

async def process(users: List[Dict], **kwargs) -> Dict:
    """Process users with AI-powered filtering."""
    config = FilterConfig(**kwargs)
    processor = FilterProcessor(config)
    return await processor.process_filters(users)

__all__ = ["FilterConfig", "FilterProcessor", "configure", "process"] 