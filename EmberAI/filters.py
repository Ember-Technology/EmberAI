"""
EmberAI.filters: AI-powered filtering and enrichment system.
"""

from typing import List, Dict
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
    block_eu: bool = False
    block_legal: bool = False
    block_eu_usernames: bool = False
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
        if self.config.block_eu:
            current_users = await self._filter_eu_users(current_users)
            filters_applied.append("block_eu")
        
        if self.config.block_legal:
            current_users = await self._filter_legal_users(current_users)
            filters_applied.append("block_legal")
        
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
        system_prompt = """<task>
  <objective>
    Detect whether the provided Twitter user is likely a resident of the European Union (EU).
  </objective>
  <input>
    For each user, you'll receive:
    <location>user location</location>
    <name>user name</name>
    <description>user bio/description</description>
  </input>
  <language_support>
    Recognize EU residency indicators in multiple languages and formats:
    Include:
    - EU country names (Germany, France, Spain, Italy, Netherlands, etc.)
    - EU cities (Berlin, Paris, Madrid, Rome, Amsterdam, Brussels, etc.)
    - EU languages in bio (German, French, Italian, Spanish, Dutch, etc.)
    - EU-specific references (GDPR, European Parliament, Euro currency €, EU timezones)
    - EU cultural references (Eurovision, European football leagues, EU holidays)
    - Educational institutions (European universities, EU exchange programs)
    - EU professional contexts (European companies, EU regulations)
  </language_support>
  <output_format>
    Return exactly one classification per user, one per line:
    - <classification>EU</classification> (if ≥2 strong indicators or 1 strong + 2 weak)
    - <classification>NOT_EU</classification> (otherwise)
  </output_format>
  <rules>
    1. Strong indicators: EU country/city names, EU languages, explicit EU references, EU timezone formats
    2. Weak indicators: European cultural references, EU-style date formats, European company names
    3. Exclude: Non-EU European countries (UK post-Brexit, Switzerland, Norway, etc.)
    4. Consider context: "visiting Paris" ≠ "living in Paris"
    5. Default to NOT_EU if ambiguous or insufficient information
    6. Ignore VPN/proxy locations without other supporting evidence
  </rules>
</task>

Users to classify:"""
        
        input_blocks = []
        for idx, user in enumerate(users, start=1):
            input_blocks.append(f"""<user_{idx}>
<location>{user.get('location', '')}</location>
<name>{user.get('name', '')}</name>
<description>{user.get('description', '')}</description>
</user_{idx}>""")
        
        return f"{system_prompt}\n\n" + "\n".join(input_blocks)
    
    def _build_legal_prompt(self, users: List[Dict]) -> str:
        """Build prompt for legal profession classification."""
        system_prompt = """<task>
  <objective>
    Detect whether the provided Twitter user is a lawyer, attorney, or legal professional.
  </objective>
  <input>
    For each user, you'll receive:
    <username>username</username>
    <full_name>full name</full_name>
    <description>bio/description</description>
  </input>
  <language_support>
    Recognize legal professions in multiple languages (English, German, French, Spanish, etc.).
    Include:
    - Job titles (attorney, lawyer, barrister, counsel, solicitor, notary, judge, prosecutor)
    - Education (JD, LLB, LLM, "Harvard Law", "Yale Law", "admitted to [bar]", "bar certified")
    - Firms (law firms, legal departments, courts, "BigLaw", "Skadden", "DLA Piper")
    - Professional indicators (Esq., "partner at", "associate at", "legal counsel")
    - Emojis (⚖️, 🧑‍⚖️), hashtags (#LawyerLife, #LegalEagle, #AttorneyLife)
    - Practice areas (litigation, corporate law, family law, criminal defense, etc.)
    - Bar associations, legal certifications, court admissions
  </language_support>
  <output_format>
    Return exactly one classification per user, one per line:
    - <classification>LEGAL</classification> (if ≥2 strong indicators or 1 strong + 2 weak)
    - <classification>NOT_LEGAL</classification> (otherwise)
  </output_format>
  <rules>
    1. Strong indicators: "attorney," "lawyer," "admitted to the bar," "law firm partner," "Esq.," "JD," specific law schools
    2. Weak indicators: "legal," "justice," law-related emojis, "LLM," "legal studies," "paralegal"
    3. Exclude: Law students (unless practicing), legal analysts without JD, compliance officers, paralegals
    4. Ignore sarcasm/parody (e.g., "not a real lawyer," "armchair lawyer")
    5. Consider context: "legal advice" in bio vs "don't take this as legal advice"
    6. Default to NOT_LEGAL if ambiguous or insufficient information
  </rules>
</task>

Users to classify:"""
        
        input_blocks = []
        for idx, user in enumerate(users, start=1):
            input_blocks.append(f"""<user_{idx}>
<username>{user.get('username', '')}</username>
<full_name>{user.get('full_name', '')}</full_name>
<description>{user.get('description', '')}</description>
</user_{idx}>""")
        
        return f"{system_prompt}\n\n" + "\n".join(input_blocks)
    
    def _build_gender_prompt(self, users: List[Dict]) -> str:
        """Build prompt for gender classification."""
        system_prompt = """<task>
  <objective>
    Classify the likely gender of Twitter users based on their names and usernames, with cultural sensitivity.
  </objective>
  <input>
    For each user, you'll receive:
    <name>full name or display name</name>
    <username>username/handle</username>
  </input>
  <language_support>
    Recognize names across multiple cultures and languages:
    Include:
    - Western names (John/Jane, Michael/Michelle, David/Diana)
    - European names (Giovanni/Giovanna, François/Françoise, Hans/Hanna)
    - Asian names (Hiroshi/Hiroko, Wei/Wei, Raj/Priya)
    - Arabic names (Mohammed/Fatima, Ahmed/Aisha)
    - African names (Kwame/Ama, Chike/Ngozi)
    - Latin American names (Carlos/Carla, José/María)
    - Gender-neutral names (Alex, Taylor, Jordan, Casey)
    - Modern/unique names and cultural variations
  </language_support>
  <output_format>
    Return exactly one classification per user, one per line:
    - <gender>male</gender> (strong indicators of male identity)
    - <gender>female</gender> (strong indicators of female identity)
    - <gender>unknown</gender> (ambiguous, gender-neutral, or insufficient information)
  </output_format>
  <rules>
    1. Prioritize first names over surnames for gender classification
    2. Consider cultural context (same name may have different gender implications across cultures)
    3. Handle compound names (Mary-Jane = female, Jean-Pierre = male)
    4. Username patterns: numbers/random characters usually indicate unknown
    5. Common gender-neutral names default to unknown unless other indicators present
    6. Respect non-binary possibilities - when in doubt, classify as unknown
    7. Don't make assumptions based on stereotypes or professions
    8. Handle initials (J. Smith) as unknown unless clear context
  </rules>
</task>

Users to classify:"""
        
        input_blocks = []
        for idx, user in enumerate(users, start=1):
            name = user.get('full_name') or user.get('name', '')
            username = user.get('username', '')
            input_blocks.append(f"""<user_{idx}>
<name>{name}</name>
<username>{username}</username>
</user_{idx}>""")
        
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