"""
EmberAI.filters: AI-powered filtering and enrichment system.
"""

from typing import List, Dict
from dataclasses import dataclass
import logging
import time
import asyncio
import os
import json
from aiolimiter import AsyncLimiter
from .ai import get_instance as get_ai_instance

logger = logging.getLogger(__name__)

UNIQUE_EU_USERNAMES = [
    "Albrecht",
    "Ansgar",
    "Bernd",
    "Björn",
    "Clemens",
    "Dieter",
    "Eberhard",
    "Eckhart",
    "Egon",
    "Ernst",
    "Falko",
    "Friedrich",
    "Gernot",
    "Gunther",
    "Hagen",
    "Hansjörg",
    "Hartmut",
    "Heiner",
    "Helmut",
    "Henning",
    "Hermann",
    "Horst",
    "Ingmar",
    "Jörg",
    "Jürgen",
    "Kai-Uwe",
    "Klaus",
    "Konrad",
    "Lutz",
    "Manfred",
    "Marcel",
    "Markus",
    "Matthias",
    "Norbert",
    "Olaf",
    "Otto",
    "Pascal",
    "Rainer",
    "Reinhard",
    "Roland",
    "Rolf",
    "Rüdiger",
    "Siegfried",
    "Steffen",
    "Thilo",
    "Thorsten",
    "Timo",
    "Ulrich",
    "Uwe",
    "Wolfgang",
    "Anneliese",
    "Babette",
    "Beate",
    "Brigitte",
    "Carina",
    "Dagmar",
    "Edeltraud",
    "Elfriede",
    "Erika",
    "Friederike",
    "Gabriele",
    "Gertrud",
    "Gudrun",
    "Heike",
    "Helga",
    "Hildegard",
    "Ilse",
    "Ingrid",
    "Irmgard",
    "Jutta",
    "Karin",
    "Katja",
    "Kirsten",
    "Lieselotte",
    "Lore",
    "Magda",
    "Margarete",
    "Marianne",
    "Marlies",
    "Monika",
    "Petra",
    "Renate",
    "Sabine",
    "Sigrid",
    "Silke",
    "Suse",
    "Sybille",
    "Thea",
    "Traudel",
    "Ulla",
    "Ursula",
    "Verena",
    "Walburga",
    "Waltraud",
    "Wilhelmine",
    "Yvonne",
    "Arno",
    "Detlef",
    "Gisela",
    "Heino",
    "Inka",
    "Lothar",
    "Ottilie",
    "Siegmund",
    "Theodor",
    "Volkmar"
]

@dataclass
class FilterConfig:
    """Configuration for AI filters with toggles."""
    block_eu: bool = False
    block_legal: bool = False
    block_unique_names: bool = False
    enrich_gender: bool = False
    preserve_order: bool = True
    log_stats: bool = True
    batch_size: int = 25
    rate_limit_per_sec: int = 50
    max_concurrent: int = 128

# Configuration via env vars
BATCH_SIZE: int = int(os.getenv("EU_BATCH_SIZE", 25))
RATE_LIMIT_PER_SEC: int = int(os.getenv("EMBER_API_RATE_PER_SEC", 50))
MAX_IN_FLIGHT: int = int(os.getenv("EU_MAX_CONCURRENT", 128))

# Rate limiter and semaphore to cap concurrency & vendor usage
_limiter = AsyncLimiter(RATE_LIMIT_PER_SEC, time_period=1)
_semaphore = asyncio.Semaphore(MAX_IN_FLIGHT)

# EU Filter System Prompts
EU_SYSTEM_PROMPT = """\
You are an EU-location classifier with optional name-based blocking.
Return exactly one XML tag:
<classification>EU</classification>
or
<classification>NOT_EU</classification>
No extra text."""

EU_FEW_SHOT = """\
<example>
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
</example>\
"""

# Legal Filter System Prompts
LEGAL_SYSTEM_PROMPT = """\
You are a legal profession classifier that identifies lawyers, attorneys, and legal professionals.

CLASSIFICATION RULES:
- LEGAL: Licensed attorneys, lawyers, judges, prosecutors, legal counsel, law firm partners/associates
- NOT_LEGAL: Law students, paralegals, legal analysts, compliance officers, legal assistants

STRONG INDICATORS (any 1 = LEGAL):
- Professional titles: Attorney, Lawyer, Esq., Counsel, Barrister, Solicitor, Judge, Prosecutor
- Law firm affiliations: Partner, Associate, Of Counsel at law firms
- Bar admissions: "Admitted to [State] Bar", "Licensed attorney"
- Legal degrees in practice: JD, LLB with current legal work
- Court roles: Judge, Magistrate, Legal Clerk

WEAK INDICATORS (need 2+ = LEGAL):
- Legal education: Law school names, LLM degree
- Legal-adjacent roles: In-house counsel, legal department
- Legal hashtags: #LawyerLife, #AttorneyLife, #LegalEagle
- Legal emojis: ⚖️, 🧑‍⚖️

EXCLUSIONS (always NOT_LEGAL):
- Law students (unless also practicing)
- Paralegals, legal assistants, legal secretaries
- Compliance officers, legal analysts without JD
- Non-practicing JDs in other fields
- Sarcasm: "armchair lawyer", "not real legal advice"

Return exactly one XML tag:
<classification>LEGAL</classification>
or
<classification>NOT_LEGAL</classification>
No extra text."""

LEGAL_FEW_SHOT = """\
<example>
  <username>john_attorney</username>
  <full_name>John Smith, Esq.</full_name>
  <description>Partner at Skadden, Arps. Corporate law.</description>
  <classification>LEGAL</classification>
</example>
<example>
  <username>judge_martinez</username>
  <full_name>Hon. Maria Martinez</full_name>
  <description>Superior Court Judge, 15th District</description>
  <classification>LEGAL</classification>
</example>
<example>
  <username>techguy</username>
  <full_name>Mike Johnson</full_name>
  <description>Software engineer at startup. Code = Law ⚖️</description>
  <classification>NOT_LEGAL</classification>
</example>
<example>
  <username>lawstudent2024</username>
  <full_name>Sarah Davis</full_name>
  <description>3L at Harvard Law School. Future BigLaw associate!</description>
  <classification>NOT_LEGAL</classification>
</example>
<example>
  <username>legalcounsel</username>
  <full_name>David Chen, JD</full_name>
  <description>General Counsel at TechCorp. Admitted NY Bar.</description>
  <classification>LEGAL</classification>
</example>
<example>
  <username>paralegal_jane</username>
  <full_name>Jane Wilson</full_name>
  <description>Paralegal at Thompson & Associates. 10 years experience.</description>
  <classification>NOT_LEGAL</classification>
</example>
<example>
  <username>prosecutor_smith</username>
  <full_name>Robert Smith</full_name>
  <description>Assistant District Attorney, Criminal Division</description>
  <classification>LEGAL</classification>
</example>
<example>
  <username>compliance_pro</username>
  <full_name>Lisa Brown</full_name>
  <description>Compliance Officer at Bank. Legal background.</description>
  <classification>NOT_LEGAL</classification>
</example>\
"""

# Gender Filter System Prompts
GENDER_SYSTEM_PROMPT = """\
You are a gender classifier based on names.
Return exactly one XML tag:
<gender>male</gender>
or
<gender>female</gender>
or
<gender>unknown</gender>
No extra text."""

GENDER_FEW_SHOT = """\
<example>
  <name>John Smith</name>
  <username>johnsmith</username>
  <gender>male</gender>
</example>
<example>
  <name>Maria Garcia</name>
  <username>maria_g</username>
  <gender>female</gender>
</example>
<example>
  <name></name>
  <username>crypto123</username>
  <gender>unknown</gender>
</example>\
"""

BATCH_OUTPUT_INSTRUCTIONS = (
    "Return exactly one classification tag for every input block, "
    "one per line, in the same order. Do not add extra text."
)

def build_eu_batch_prompt(users: List[Dict], block_unique_names: bool = False) -> str:
    """Build EU classification batch prompt."""
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
    if block_unique_names:
        inputs_section += "\n" + "\n".join(UNIQUE_EU_USERNAMES)
    
    return (
        f"SYSTEM:\n{EU_SYSTEM_PROMPT}\n\n"
        f"Few-shot examples:\n{EU_FEW_SHOT}\n\n"
        f"Inputs:\n{inputs_section}\n\n"
        f"Output:\n{BATCH_OUTPUT_INSTRUCTIONS}"
    )

def build_legal_batch_prompt(users: List[Dict]) -> str:
    """Build legal classification batch prompt."""
    input_blocks = []
    for idx, user in enumerate(users, start=1):
        input_blocks.append(
            f"""<input index=\"{idx}\">
  <username>{user.get('username', '')}</username>
  <full_name>{user.get('full_name', '')}</full_name>
  <description>{user.get('description', '')}</description>
</input>"""
        )
    
    inputs_section = "\n".join(input_blocks)
    
    return (
        f"SYSTEM:\n{LEGAL_SYSTEM_PROMPT}\n\n"
        f"Few-shot examples:\n{LEGAL_FEW_SHOT}\n\n"
        f"Inputs:\n{inputs_section}\n\n"
        f"Output:\n{BATCH_OUTPUT_INSTRUCTIONS}"
    )

def build_gender_batch_prompt(users: List[Dict]) -> str:
    """Build gender classification batch prompt."""
    input_blocks = []
    for idx, user in enumerate(users, start=1):
        name = user.get('full_name') or user.get('name', '')
        username = user.get('username', '')
        input_blocks.append(
            f"""<input index=\"{idx}\">
  <name>{name}</name>
  <username>{username}</username>
</input>"""
        )
    
    inputs_section = "\n".join(input_blocks)
    
    return (
        f"SYSTEM:\n{GENDER_SYSTEM_PROMPT}\n\n"
        f"Few-shot examples:\n{GENDER_FEW_SHOT}\n\n"
        f"Inputs:\n{inputs_section}\n\n"
        f"Output:\n{BATCH_OUTPUT_INSTRUCTIONS}"
    )

def parse_eu_batch_response(response_text: str, expected: int) -> List[bool]:
    """Parse EU classification response."""
    if not response_text:
        raise ValueError("Empty response from model while parsing batch EU classifications")
    
    lines = [line.strip() for line in response_text.splitlines() if line.strip()]
    classifications: List[bool] = []
    
    for line in lines:
        if line == "<classification>EU</classification>":
            classifications.append(True)
        elif line == "<classification>NOT_EU</classification>":
            classifications.append(False)
    
    if len(classifications) != expected:
        raise ValueError(
            f"Expected {expected} classifications but parsed {len(classifications)} "
            "from model response.\nResponse was:\n" + response_text
        )
    return classifications

def parse_legal_batch_response(response_text: str, expected: int) -> List[bool]:
    """Parse legal classification response."""
    if not response_text:
        raise ValueError("Empty response from model while parsing batch legal classifications")
    
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
            "from model response.\nResponse was:\n" + response_text
        )
    return classifications

def parse_gender_batch_response(response_text: str, expected: int) -> List[str]:
    """Parse gender classification response."""
    if not response_text:
        raise ValueError("Empty response from model while parsing batch gender classifications")
    
    lines = [line.strip() for line in response_text.splitlines() if line.strip()]
    genders: List[str] = []
    
    for line in lines:
        if line.startswith("<gender>") and line.endswith("</gender>"):
            gender = line[8:-9].strip().lower()
            genders.append(gender if gender in ["male", "female", "unknown"] else "unknown")
    
    if len(genders) != expected:
        raise ValueError(
            f"Expected {expected} gender classifications but parsed {len(genders)} "
            "from model response.\nResponse was:\n" + response_text
        )
    return genders

def classify_eu_batch(users: List[Dict], block_unique_names: bool = False) -> List[bool]:
    """Classify a batch of users for EU filtering."""
    prompt = build_eu_batch_prompt(users, block_unique_names)
    response = get_ai_instance().process(prompt)
    return parse_eu_batch_response(response, expected=len(users))

def classify_legal_batch(users: List[Dict]) -> List[bool]:
    """Classify a batch of users for legal filtering."""
    prompt = build_legal_batch_prompt(users)
    response = get_ai_instance().process(prompt)
    return parse_legal_batch_response(response, expected=len(users))

def classify_gender_batch(users: List[Dict]) -> List[str]:
    """Classify a batch of users for gender enrichment."""
    prompt = build_gender_batch_prompt(users)
    response = get_ai_instance().process(prompt)
    return parse_gender_batch_response(response, expected=len(users))

async def _classify_eu_single_batch_async(batch: List[Dict], block_unique_names: bool) -> List[bool]:
    """Run one EU batch inside limiter + semaphore in an executor."""
    async with _semaphore, _limiter:
        return await asyncio.to_thread(classify_eu_batch, batch, block_unique_names)

async def _classify_legal_single_batch_async(batch: List[Dict]) -> List[bool]:
    """Run one legal batch inside limiter + semaphore in an executor."""
    async with _semaphore, _limiter:
        return await asyncio.to_thread(classify_legal_batch, batch)

async def _classify_gender_single_batch_async(batch: List[Dict]) -> List[str]:
    """Run one gender batch inside limiter + semaphore in an executor."""
    async with _semaphore, _limiter:
        return await asyncio.to_thread(classify_gender_batch, batch)

async def classify_eu_users_async(users: List[Dict], block_unique_names: bool = False, batch_size: int = BATCH_SIZE) -> List[bool]:
    """High-throughput EU classification preserving order."""
    if not users:
        return []
    
    batches: List[List[Dict]] = [
        users[i : i + batch_size] for i in range(0, len(users), batch_size)
    ]
    
    tasks = [
        asyncio.create_task(_classify_eu_single_batch_async(batch, block_unique_names))
        for batch in batches
    ]
    
    results: List[bool] = []
    for batch_res in await asyncio.gather(*tasks):
        results.extend(batch_res)
    return results

async def classify_legal_users_async(users: List[Dict], batch_size: int = BATCH_SIZE) -> List[bool]:
    """High-throughput legal classification preserving order."""
    if not users:
        return []
    
    batches: List[List[Dict]] = [
        users[i : i + batch_size] for i in range(0, len(users), batch_size)
    ]
    
    tasks = [
        asyncio.create_task(_classify_legal_single_batch_async(batch))
        for batch in batches
    ]
    
    results: List[bool] = []
    for batch_res in await asyncio.gather(*tasks):
        results.extend(batch_res)
    return results

async def classify_gender_users_async(users: List[Dict], batch_size: int = BATCH_SIZE) -> List[str]:
    """High-throughput gender classification preserving order."""
    if not users:
        return []
    
    batches: List[List[Dict]] = [
        users[i : i + batch_size] for i in range(0, len(users), batch_size)
    ]
    
    tasks = [
        asyncio.create_task(_classify_gender_single_batch_async(batch))
        for batch in batches
    ]
    
    results: List[str] = []
    for batch_res in await asyncio.gather(*tasks):
        results.extend(batch_res)
    return results

# Global configuration
_config = FilterConfig()

def configure(**kwargs):
    """Configure the filters module."""
    global _config
    for key, value in kwargs.items():
        if hasattr(_config, key):
            setattr(_config, key, value)
    logger.info("✅ EmberAI Filters configured")

async def process(users: List[Dict], **kwargs) -> List[Dict]:
    """Process users with AI-powered filtering."""
    if not users:
        return []
    
    ai_processor = get_ai_instance()
    if not ai_processor.is_configured():
        raise RuntimeError("EmberAI not configured. Call EmberAI.ai.configure() first.")
    
    config = FilterConfig(**kwargs)
    start_time = time.time()
    current_users = users.copy() if config.preserve_order else users
    filters_applied = []
    
    # Apply EU filter
    if config.block_eu:
        eu_classifications = await classify_eu_users_async(
            current_users, 
            block_unique_names=config.block_unique_names,
            batch_size=config.batch_size
        )
        current_users = [u for u, is_eu in zip(current_users, eu_classifications) if not is_eu]
        filters_applied.append("block_eu")
        if config.block_unique_names:
            filters_applied.append("block_unique_names")
    
    # Apply legal filter
    if config.block_legal:
        legal_classifications = await classify_legal_users_async(
            current_users,
            batch_size=config.batch_size
        )
        current_users = [u for u, is_legal in zip(current_users, legal_classifications) if not is_legal]
        filters_applied.append("block_legal")
    
    # Apply gender enrichment
    if config.enrich_gender:
        gender_classifications = await classify_gender_users_async(
            current_users,
            batch_size=config.batch_size
        )
        enriched_users = []
        for user, gender in zip(current_users, gender_classifications):
            enriched_user = dict(user)
            enriched_user['gender'] = gender
            enriched_users.append(enriched_user)
        current_users = enriched_users
        filters_applied.append("enrich_gender")
    
    total_time = time.time() - start_time
    
    if config.log_stats:
        logger.info(f"✅ Filters applied: {filters_applied}")
        logger.info(f"✅ Original count: {len(users)}")
        logger.info(f"✅ Final count: {len(current_users)}")
        logger.info(f"✅ Processing time: {total_time:.2f}s")
        logger.info(f"✅ Processing rate: {len(users) / total_time if total_time > 0 else 0:.1f} users/sec")
    
    return current_users

__all__ = [
    "FilterConfig", 
    "configure", 
    "process",
    "build_eu_batch_prompt",
    "parse_eu_batch_response", 
    "classify_eu_batch",
    "classify_eu_users_async",
    "build_legal_batch_prompt",
    "parse_legal_batch_response",
    "classify_legal_batch", 
    "classify_legal_users_async",
    "build_gender_batch_prompt",
    "parse_gender_batch_response",
    "classify_gender_batch",
    "classify_gender_users_async"
] 