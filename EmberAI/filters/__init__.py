"""
EmberAI.filters: AI-powered filtering and enrichment system.

This module provides high-performance filtering and enrichment capabilities
using the EmberAI processing engine.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import logging
import time
import asyncio
from aiolimiter import AsyncLimiter
from ..ai import get_instance as get_ai_instance

logger = logging.getLogger(__name__)

@dataclass
class FilterConfig:
    """Configuration for AI filters with toggles."""
    # Blocking filters (remove users)
    block_eu_users: bool = False
    block_legal_users: bool = False
    block_unique_names: bool = False  # For EU filter
    
    # Enrichment filters (add data)
    enrich_gender: bool = False
    
    # Processing options
    preserve_order: bool = True
    log_stats: bool = True
    
    # Performance tuning
    batch_size: int = 25
    rate_limit_per_sec: int = 50
    max_concurrent: int = 128

class FilterProcessor:
    """
    Self-contained AI filtering system with all filtering and enrichment logic.
    
    Features:
    - High-throughput async processing
    - Configurable filter toggles
    - EU location detection and blocking
    - Legal professional detection and blocking
    - Gender detection and enrichment
    - Comprehensive logging and stats
    - Integrated with EmberAI processing engine
    """
    
    # EU Names for location-based blocking
    EU_NAMES = [
        "Müller", "Schmidt", "Schneider", "Fischer", "Weber", "Meyer", "Wagner", "Becker",
        "Schulz", "Hoffmann", "Schäfer", "García", "González", "Rodríguez", "Fernández",
        "López", "Martínez", "Sánchez", "Pérez", "Gómez", "Martin", "Bernard", "Dubois",
        "Thomas", "Robert", "Petit", "Durand", "Leroy", "Moreau", "Simon", "Laurent",
        "Lefebvre", "Michel", "Garcia", "David", "Bertrand", "Roux", "Vincent", "Fournier",
        "Rossi", "Russo", "Ferrari", "Esposito", "Bianchi", "Romano", "Colombo", "Ricci",
        "Marino", "Greco", "Bruno", "Gallo", "Conti", "De Luca", "Mancini", "Costa",
        "Nielsen", "Hansen", "Larsen", "Andersen", "Pedersen", "Nilsson", "Petersen",
        "Andersson", "Lindqvist", "Olsson", "Persson", "Svensson", "Gustafsson", "Jonsson",
        "Karlsson", "Larsson", "Nilsson", "Petersson", "Samuelsson", "Wikström"
    ]
    
    def __init__(self, config: FilterConfig):
        self.config = config
        self._setup_rate_limiting()
    
    def _setup_rate_limiting(self):
        """Initialize rate limiting and concurrency controls."""
        self._limiter = AsyncLimiter(self.config.rate_limit_per_sec, time_period=1)
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)
    
    # ==========================================
    # EU LOCATION DETECTION AND BLOCKING
    # ==========================================
    
    def _build_eu_batch_prompt(self, users: List[Dict[str, str]], block_unique_names: bool = False) -> str:
        """Build batch prompt for EU location detection."""
        system_prompt = """You are an EU-location classifier with optional name-based blocking.
Return exactly one XML tag:
<classification>EU</classification>
or
<classification>NOT_EU</classification>
No extra text."""

        few_shot = """<example>
  <location>Berlin</location>
  <description>Tech enthusiast in Mitte</description>
  <name>Johannes Schmidt</name>
  <classification>EU</classification>
</example>
<example>
  <location>New York, USA</location>
  <description>Librarian</description>
  <name>Emma Johnson</name>
  <classification>NOT_EU</classification>
</example>
<example>
  <location></location>
  <description></description>
  <name>cryptoFan123</name>
  <classification>NOT_EU</classification>
</example>
<example>
  <location></location>
  <description></description>
  <name>Eckhart Müller</name>
  <classification>EU</classification>
</example>"""

        toggle = "true" if block_unique_names else "false"
        
        input_blocks = []
        for idx, user in enumerate(users, start=1):
            input_blocks.append(
                f"""<input index=\"{idx}\">
  <location>{user.get('location', '')}</location>
  <description>{user.get('description', '')}</description>
  <name>{user.get('name', '')}</name>
  <block_unique_names>{toggle}</block_unique_names>
</input>"""
            )

        inputs_section = "\n".join(input_blocks)
        inputs_section += "\n" + "\n".join(self.EU_NAMES)

        return (
            f"SYSTEM:\n{system_prompt}\n\n"
            f"Few-shot examples:\n{few_shot}\n\n"
            f"Inputs:\n{inputs_section}\n\n"
            f"Output:\nReturn exactly one <classification/> tag for every <input/> block, "
            "one per line, in the same order. Do not add extra text."
        )
    
    def _parse_eu_batch_response(self, response_text: str, expected: int) -> List[bool]:
        """Parse EU classification response into booleans."""
        if not response_text:
            raise ValueError("Empty response from EU classifier")

        lines = [line.strip() for line in response_text.splitlines() if line.strip()]
        classifications: List[bool] = []
        
        for line in lines:
            if line == "<classification>EU</classification>":
                classifications.append(True)
            elif line == "<classification>NOT_EU</classification>":
                classifications.append(False)

        if len(classifications) != expected:
            raise ValueError(
                f"Expected {expected} EU classifications but parsed {len(classifications)} "
                f"from response: {response_text}"
            )
        return classifications
    
    def _classify_eu_batch(self, users: List[Dict[str, str]], block_unique_names: bool = False) -> List[bool]:
        """Synchronous EU batch classification."""
        prompt = self._build_eu_batch_prompt(users, block_unique_names)
        ai_processor = get_ai_instance()
        response = ai_processor.process(prompt)
        return self._parse_eu_batch_response(response, expected=len(users))
    
    async def _classify_eu_batch_async(self, batch: List[Dict[str, str]], block_unique_names: bool) -> List[bool]:
        """Async EU batch classification with rate limiting."""
        async with self._semaphore, self._limiter:
            return await asyncio.to_thread(self._classify_eu_batch, batch, block_unique_names)
    
    async def _classify_eu_users_async(self, users: List[Dict[str, str]], block_unique_names: bool = False) -> List[bool]:
        """High-throughput EU classification."""
        if not users:
            return []

        batches = [users[i:i + self.config.batch_size] for i in range(0, len(users), self.config.batch_size)]
        tasks = [asyncio.create_task(self._classify_eu_batch_async(batch, block_unique_names)) for batch in batches]

        results: List[bool] = []
        for batch_res in await asyncio.gather(*tasks):
            results.extend(batch_res)
        return results
    
    async def _filter_eu_users(self, users: List[Dict[str, str]], block_unique_names: bool = False) -> List[Dict[str, str]]:
        """Filter out EU users."""
        classifications = await self._classify_eu_users_async(users, block_unique_names)
        return [u for u, is_eu in zip(users, classifications) if not is_eu]
    
    # ==========================================
    # LEGAL PROFESSION DETECTION AND BLOCKING
    # ==========================================
    
    def _build_legal_batch_prompt(self, users: List[Dict[str, str]]) -> str:
        """Build batch prompt for legal profession detection."""
        system_prompt = """You are a legal professional classifier.
For every <input/> block, analyze the provided user information and determine if they are a lawyer, attorney, or legal professional.
Return exactly one XML tag:
<classification>LEGAL</classification>
or
<classification>NOT_LEGAL</classification>
No extra text."""

        few_shot = """<example>
  <username>johnsmith_esq</username>
  <full_name>John Smith</full_name>
  <description>Partner at Skadden Arps. Harvard Law '95. Admitted to NY Bar.</description>
  <education>Harvard Law School</education>
  <employer>Skadden Arps</employer>
  <classification>LEGAL</classification>
</example>
<example>
  <username>techguru</username>
  <full_name>Sarah Johnson</full_name>
  <description>Software engineer at Google. Love coding and coffee ☕</description>
  <education>MIT Computer Science</education>
  <employer>Google</employer>
  <classification>NOT_LEGAL</classification>
</example>
<example>
  <username>attorney_mike</username>
  <full_name>Michael Brown</full_name>
  <description>Criminal defense attorney ⚖️ Fighting for justice</description>
  <education></education>
  <employer></employer>
  <classification>LEGAL</classification>
</example>
<example>
  <username>randomuser123</username>
  <full_name></full_name>
  <description>Just a regular person living life</description>
  <education></education>
  <employer></employer>
  <classification>NOT_LEGAL</classification>
</example>"""

        input_blocks = []
        for idx, user in enumerate(users, start=1):
            input_blocks.append(
                f"""<input index=\"{idx}\">
  <username>{user.get('username', '')}</username>
  <full_name>{user.get('full_name', '')}</full_name>
  <description>{user.get('description', '')}</description>
  <education>{user.get('education', '')}</education>
  <employer>{user.get('employer', '')}</employer>
</input>"""
            )

        inputs_section = "\n".join(input_blocks)

        return (
            f"SYSTEM:\n{system_prompt}\n\n"
            f"Few-shot examples:\n{few_shot}\n\n"
            f"Inputs:\n{inputs_section}\n\n"
            f"Output:\nReturn exactly one <classification/> tag for every <input/> block, "
            "one per line, in the same order. Do not add extra text."
        )
    
    def _parse_legal_batch_response(self, response_text: str, expected: int) -> List[bool]:
        """Parse legal classification response into booleans."""
        if not response_text:
            raise ValueError("Empty response from legal classifier")

        lines = [line.strip() for line in response_text.splitlines() if line.strip()]
        classifications: List[bool] = []
        
        for line in lines:
            if line == "<classification>LEGAL</classification>":
                classifications.append(True)
            elif line == "<classification>NOT_LEGAL</classification>":
                classifications.append(False)

        if len(classifications) != expected:
            raise ValueError(
                f"Expected {expected} legal classifications but parsed {len(classifications)} "
                f"from response: {response_text}"
            )
        return classifications
    
    def _classify_legal_batch(self, users: List[Dict[str, str]]) -> List[bool]:
        """Synchronous legal batch classification."""
        prompt = self._build_legal_batch_prompt(users)
        ai_processor = get_ai_instance()
        response = ai_processor.process(prompt)
        return self._parse_legal_batch_response(response, expected=len(users))
    
    async def _classify_legal_batch_async(self, batch: List[Dict[str, str]]) -> List[bool]:
        """Async legal batch classification with rate limiting."""
        async with self._semaphore, self._limiter:
            return await asyncio.to_thread(self._classify_legal_batch, batch)
    
    async def _classify_legal_users_async(self, users: List[Dict[str, str]]) -> List[bool]:
        """High-throughput legal classification."""
        if not users:
            return []

        batches = [users[i:i + self.config.batch_size] for i in range(0, len(users), self.config.batch_size)]
        tasks = [asyncio.create_task(self._classify_legal_batch_async(batch)) for batch in batches]

        results: List[bool] = []
        for batch_res in await asyncio.gather(*tasks):
            results.extend(batch_res)
        return results
    
    async def _block_legal_users(self, users: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Block/filter out legal professionals."""
        classifications = await self._classify_legal_users_async(users)
        return [u for u, is_legal in zip(users, classifications) if not is_legal]
    
    # ==========================================
    # GENDER DETECTION AND ENRICHMENT
    # ==========================================
    
    def _build_gender_batch_prompt(self, users: List[Dict[str, str]]) -> str:
        """Build batch prompt for gender detection."""
        system_prompt = """You are a **gender classifier**.
For every <input/> block, analyse the provided <name/> and <username/> fields and
decide whether the person is <gender>male</gender>, <gender>female</gender>, or
<gender>unknown</gender>.

Guidelines:
• Prefer the full name if present; otherwise extract hints from the username.
• If the string looks like a brand / organisation (e.g. "NikeOfficial"), or if
  the gender is ambiguous, return <gender>unknown</gender>.
• Return EXACTLY one gender XML tag per input, no commentary."""

        few_shot = """<example>
  <name>John Doe</name>
  <username>johndoe</username>
  <gender>male</gender>
</example>
<example>
  <name>Ayesha Khan</name>
  <username>ayesha_k</username>
  <gender>female</gender>
</example>
<example>
  <name></name>
  <username>CryptoKing123</username>
  <gender>unknown</gender>
</example>"""

        input_blocks: List[str] = []
        for idx, user in enumerate(users, start=1):
            name = user.get("full_name") or user.get("name", "")
            username = user.get("username", "")
            input_blocks.append(
                f"""<input index=\"{idx}\">
  <name>{name}</name>
  <username>{username}</username>
</input>"""
            )
        inputs_section = "\n".join(input_blocks)
        
        return (
            f"SYSTEM:\n{system_prompt}\n\n"
            f"Few-shot examples:\n{few_shot}\n\n"
            f"Inputs:\n{inputs_section}\n\n"
            f"Output:\nReturn one <gender/> tag for every <input/> block, in the same order, "
            "each on its own line. No extra text."
        )
    
    def _parse_gender_batch_response(self, response_text: str, expected: int) -> List[str]:
        """Parse gender response into list of gender strings."""
        if not response_text:
            raise ValueError("Empty response from gender classifier")

        lines = [line.strip() for line in response_text.splitlines() if line.strip()]
        genders: List[str] = []
        
        for line in lines:
            if line.startswith("<gender>") and line.endswith("</gender>"):
                genders.append(line[8:-9].strip().lower())
                
        if len(genders) != expected:
            raise ValueError(
                f"Expected {expected} gender tags, got {len(genders)}. Response: {response_text}"
            )
        return genders
    
    def _validate_gender_value(self, gender_value: str) -> str:
        """Ensure gender value is always one of the three allowed values."""
        if not gender_value or not isinstance(gender_value, str):
            return "unknown"
        
        gender_clean = gender_value.strip().lower()
        
        if gender_clean in ["male", "female", "unknown"]:
            return gender_clean
        
        # Handle common variations
        male_variants = ["m", "man", "masculine", "boy", "gentleman"]
        female_variants = ["f", "woman", "feminine", "girl", "lady"]
        unknown_variants = ["ambiguous", "unclear", "uncertain", "neutral", "other", "n/a", "na", "none", ""]
        
        if gender_clean in male_variants:
            return "male"
        elif gender_clean in female_variants:
            return "female"
        elif gender_clean in unknown_variants:
            return "unknown"
        else:
            return "unknown"
    
    def _classify_gender_batch(self, users: List[Dict[str, str]]) -> List[str]:
        """Synchronous gender batch classification."""
        prompt = self._build_gender_batch_prompt(users)
        ai_processor = get_ai_instance()
        response = ai_processor.process(prompt)
        return self._parse_gender_batch_response(response, expected=len(users))
    
    async def _classify_gender_batch_async(self, batch: List[Dict[str, str]]) -> List[str]:
        """Async gender batch classification with rate limiting."""
        async with self._semaphore, self._limiter:
            return await asyncio.to_thread(self._classify_gender_batch, batch)
    
    async def _classify_gender_users_async(self, users: List[Dict[str, str]]) -> List[str]:
        """High-throughput gender classification."""
        if not users:
            return []

        batches = [users[i:i + self.config.batch_size] for i in range(0, len(users), self.config.batch_size)]
        tasks = [asyncio.create_task(self._classify_gender_batch_async(batch)) for batch in batches]

        results: List[str] = []
        for batch_res in await asyncio.gather(*tasks):
            results.extend(batch_res)
        return results
    
    async def _enrich_gender_for_users(self, users: List[Dict]) -> List[Dict]:
        """Enrich users with gender information."""
        if not users:
            return []

        raw_genders: List[str] = await self._classify_gender_users_async(users)
        
        processed_users: List[Dict] = []
        for user, g in zip(users, raw_genders):
            processed_users.append(dict(user, gender=self._validate_gender_value(g)))

        return processed_users
    
    # ==========================================
    # MAIN PROCESSING METHOD
    # ==========================================
    
    async def process_filters(
        self, 
        users: List[Dict], 
        config_override: Optional[FilterConfig] = None
    ) -> Dict:
        """
        Apply all enabled filters and enrichments to the user list.
        
        Args:
            users: List of user dictionaries to process
            config_override: Optional config to override instance config
            
        Returns:
            Dict with processed_users, stats, and metadata
        """
        # Check if AI is configured
        ai_processor = get_ai_instance()
        if not ai_processor.is_configured():
            raise RuntimeError(
                "EmberAI not configured. Call EmberAI.ai.configure(GEMINI_API_KEY='your-key') first."
            )
        
        if not users:
            return {
                'processed_users': [],
                'original_count': 0,
                'final_count': 0,
                'processing_stats': {},
                'filters_applied': []
            }
        
        # Use override config if provided
        active_config = config_override or self.config
        
        start_time = time.time()
        original_count = len(users)
        current_users = users.copy() if active_config.preserve_order else users
        filters_applied = []
        processing_stats = {}
        
        if active_config.log_stats:
            logger.info(f"🚀 EmberAI Filters processing: {original_count:,} users")
        
        # Phase 1: Blocking filters (apply in logical order)
        if active_config.block_eu_users:
            filter_start = time.time()
            pre_eu_count = len(current_users)
            current_users = await self._filter_eu_users(
                current_users, 
                active_config.block_unique_names
            )
            filter_time = time.time() - filter_start
            blocked_count = pre_eu_count - len(current_users)
            
            filters_applied.append("block_eu_users")
            processing_stats["eu_filter"] = {
                'time_seconds': filter_time,
                'blocked_count': blocked_count,
                'remaining_count': len(current_users)
            }
            
            if active_config.log_stats:
                logger.info(f"🌍 EU Filter: Blocked {blocked_count:,} users, {len(current_users):,} remaining")
        
        if active_config.block_legal_users:
            filter_start = time.time()
            pre_legal_count = len(current_users)
            current_users = await self._block_legal_users(current_users)
            filter_time = time.time() - filter_start
            blocked_count = pre_legal_count - len(current_users)
            
            filters_applied.append("block_legal_users")
            processing_stats["legal_filter"] = {
                'time_seconds': filter_time,
                'blocked_count': blocked_count,
                'remaining_count': len(current_users)
            }
            
            if active_config.log_stats:
                logger.info(f"⚖️ Legal Filter: Blocked {blocked_count:,} users, {len(current_users):,} remaining")
        
        # Phase 2: Enrichment filters (add data to remaining users)
        if active_config.enrich_gender and current_users:
            filter_start = time.time()
            current_users = await self._enrich_gender_for_users(current_users)
            filter_time = time.time() - filter_start
            
            filters_applied.append("enrich_gender")
            processing_stats["gender_enrichment"] = {
                'time_seconds': filter_time,
                'enriched_count': len(current_users)
            }
            
            if active_config.log_stats:
                logger.info(f"👥 Gender Enrichment: Processed {len(current_users):,} users")
        
        # Final stats
        total_time = time.time() - start_time
        final_count = len(current_users)
        total_blocked = original_count - final_count
        
        processing_stats["overall"] = {
            'total_time_seconds': total_time,
            'original_count': original_count,
            'final_count': final_count,
            'total_blocked': total_blocked,
            'processing_rate': original_count / total_time if total_time > 0 else 0
        }
        
        if active_config.log_stats:
            logger.info(
                f"✅ EmberAI Filters completed: {original_count:,} → {final_count:,} users "
                f"({total_blocked:,} blocked) in {total_time:.1f}s | "
                f"Rate: {original_count/total_time:.1f}/sec"
            )
        
        return {
            'processed_users': current_users,
            'original_count': original_count,
            'final_count': final_count,
            'filters_applied': filters_applied,
            'processing_stats': processing_stats
        }

# Global configuration and instance
_config = FilterConfig()
_processor = None

def configure(**kwargs):
    """
    Configure the filters module with custom settings.
    
    Args:
        **kwargs: FilterConfig parameters
    """
    global _config, _processor
    
    # Update configuration
    for key, value in kwargs.items():
        if hasattr(_config, key):
            setattr(_config, key, value)
        else:
            logger.warning(f"Unknown filter configuration: {key}")
    
    # Create new processor with updated config
    _processor = FilterProcessor(_config)
    logger.info("✅ EmberAI Filters configured successfully")

def get_config() -> FilterConfig:
    """Get current filter configuration."""
    return _config

async def process(
    users: List[Dict],
    block_eu: bool = False,
    block_legal: bool = False,
    enrich_gender: bool = False,
    block_unique_names: bool = False,
    **kwargs
) -> Dict:
    """
    Process users with AI-powered filtering and enrichment.
    
    Args:
        users: List of user dictionaries to process
        block_eu: Block users detected as EU residents
        block_legal: Block users detected as legal professionals
        enrich_gender: Add gender information to user records
        block_unique_names: Enhanced EU blocking using name patterns
        **kwargs: Additional FilterConfig parameters
        
    Returns:
        Dict with processed_users, stats, and metadata
    """
    # Create config for this processing run
    config = FilterConfig(
        block_eu_users=block_eu,
        block_legal_users=block_legal,
        enrich_gender=enrich_gender,
        block_unique_names=block_unique_names,
        **kwargs
    )
    
    # Create processor
    processor = FilterProcessor(config)
    
    # Process users
    return await processor.process_filters(users)

def get_processor() -> Optional[FilterProcessor]:
    """Get the global filter processor instance."""
    return _processor

__all__ = [
    "FilterConfig",
    "FilterProcessor",
    "configure",
    "get_config",
    "process",
    "get_processor",
] 