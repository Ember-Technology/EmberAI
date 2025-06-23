"""
Basic usage examples for EmberAI package.

This script demonstrates how to use EmberAI for both general AI processing
and AI-powered filtering and enrichment.
"""

import asyncio
import os
import EmberAI

# Sample user data for testing
SAMPLE_USERS = [
    {
        "username": "john_attorney",
        "full_name": "John Smith",
        "description": "Partner at BigLaw Firm. Harvard Law '95. Admitted to NY Bar.",
        "location": "Berlin, Germany",
        "education": "Harvard Law School",
        "employer": "Skadden Arps"
    },
    {
        "username": "sarah_dev",
        "full_name": "Sarah Johnson",
        "description": "Software engineer at Google. Love coding and coffee ☕",
        "location": "San Francisco, CA",
        "education": "MIT Computer Science",
        "employer": "Google"
    },
    {
        "username": "alex_designer",
        "full_name": "Alex Thompson",
        "description": "UX designer passionate about creating beautiful interfaces",
        "location": "London, UK",
        "education": "Royal College of Art",
        "employer": "Design Studio"
    },
    {
        "username": "crypto_lawyer",
        "full_name": "Maria Rodriguez",
        "description": "Cryptocurrency attorney specializing in regulatory compliance",
        "location": "Madrid, Spain",
        "education": "Universidad Complutense Madrid Law",
        "employer": "Legal Tech Firm"
    }
]

def example_ai_configuration():
    """Example: Configure EmberAI with Gemini API."""
    print("=== AI Configuration Example ===")
    
    # Get API key from environment or set directly
    api_key = os.getenv("GEMINI_API_KEY", "your-gemini-api-key-here")
    
    if api_key == "your-gemini-api-key-here":
        print("⚠️  Please set your GEMINI_API_KEY environment variable")
        print("   export GEMINI_API_KEY='your-actual-api-key'")
        return False
    
    try:
        # Basic configuration
        EmberAI.ai.configure(GEMINI_API_KEY=api_key)
        print("✅ EmberAI configured successfully")
        
        # Advanced configuration with custom settings
        EmberAI.ai.configure(
            GEMINI_API_KEY=api_key,
            model="gemini-1.5-flash",
            temperature=0.1,
            max_output_tokens=1000
        )
        print("✅ Advanced configuration applied")
        return True
        
    except Exception as e:
        print(f"❌ Configuration failed: {e}")
        return False

def example_general_ai_processing():
    """Example: Use EmberAI for general AI processing."""
    print("\n=== General AI Processing Example ===")
    
    if not EmberAI.ai.is_configured():
        print("❌ EmberAI not configured. Run configuration example first.")
        return
    
    try:
        # Simple question answering
        response = EmberAI.ai.process("What is the capital of France?")
        print(f"Q: What is the capital of France?")
        print(f"A: {response}")
        
        # Creative writing with parameters
        response = EmberAI.ai.process(
            "Write a haiku about programming",
            temperature=0.7
        )
        print(f"\nHaiku about programming:")
        print(response)
        
        # Technical explanation
        response = EmberAI.ai.process(
            "Explain async/await in Python in one sentence",
            temperature=0.1
        )
        print(f"\nAsync/await explanation:")
        print(response)
        
    except Exception as e:
        print(f"❌ AI processing failed: {e}")

async def example_basic_filtering():
    """Example: Basic filtering with EmberAI.filters."""
    print("\n=== Basic Filtering Example ===")
    
    if not EmberAI.ai.is_configured():
        print("❌ EmberAI not configured. Cannot perform filtering.")
        return
    
    try:
        # Process users with EU blocking
        result = await EmberAI.filters.process(
            SAMPLE_USERS,
            block_eu_users=True
        )
        
        print(f"Original users: {result['original_count']}")
        print(f"After EU filtering: {result['final_count']}")
        print(f"Filters applied: {result['filters_applied']}")
        
        print("\nRemaining users:")
        for user in result['processed_users']:
            print(f"  - {user['full_name']} ({user['location']})")
            
    except Exception as e:
        print(f"❌ Basic filtering failed: {e}")

async def example_legal_filtering():
    """Example: Legal professional filtering."""
    print("\n=== Legal Professional Filtering Example ===")
    
    if not EmberAI.ai.is_configured():
        print("❌ EmberAI not configured. Cannot perform filtering.")
        return
    
    try:
        # Process users with legal professional blocking
        result = await EmberAI.filters.process(
            SAMPLE_USERS,
            block_legal_users=True
        )
        
        print(f"Original users: {result['original_count']}")
        print(f"After legal filtering: {result['final_count']}")
        
        print("\nNon-legal users:")
        for user in result['processed_users']:
            print(f"  - {user['full_name']}: {user['description']}")
            
    except Exception as e:
        print(f"❌ Legal filtering failed: {e}")

async def example_gender_enrichment():
    """Example: Gender enrichment."""
    print("\n=== Gender Enrichment Example ===")
    
    if not EmberAI.ai.is_configured():
        print("❌ EmberAI not configured. Cannot perform enrichment.")
        return
    
    try:
        # Process users with gender enrichment
        result = await EmberAI.filters.process(
            SAMPLE_USERS,
            enrich_gender=True
        )
        
        print(f"Enriched {result['final_count']} users with gender information:")
        for user in result['processed_users']:
            gender = user.get('gender', 'unknown')
            print(f"  - {user['full_name']}: {gender}")
            
    except Exception as e:
        print(f"❌ Gender enrichment failed: {e}")

async def example_combined_processing():
    """Example: Combined filtering and enrichment."""
    print("\n=== Combined Processing Example ===")
    
    if not EmberAI.ai.is_configured():
        print("❌ EmberAI not configured. Cannot perform processing.")
        return
    
    try:
        # Apply multiple filters and enrichments
        result = await EmberAI.filters.process(
            SAMPLE_USERS,
            block_eu_users=True,       # Remove EU users
            block_legal_users=True,    # Remove legal professionals
            enrich_gender=True         # Add gender information
        )
        
        print(f"Processing pipeline:")
        print(f"  Original users: {result['original_count']}")
        print(f"  Final users: {result['final_count']}")
        print(f"  Applied filters: {', '.join(result['filters_applied'])}")
        print(f"  Processing time: {result['processing_stats']['total_time_seconds']:.2f}s")
        
        print("\nFinal user list:")
        for user in result['processed_users']:
            gender = user.get('gender', 'unknown')
            print(f"  - {user['full_name']} ({gender}) - {user['location']}")
            
    except Exception as e:
        print(f"❌ Combined processing failed: {e}")

def example_filter_configuration():
    """Example: Custom filter configuration."""
    print("\n=== Filter Configuration Example ===")
    
    # Configure filters with custom settings
    EmberAI.filters.configure(
        batch_size=10,
        rate_limit_per_sec=20,
        max_concurrent=64,
        log_stats=True
    )
    
    print("✅ Filters configured with custom settings:")
    print("  - Batch size: 10")
    print("  - Rate limit: 20/sec")
    print("  - Max concurrent: 64")
    print("  - Logging: enabled")

async def main():
    """Run all examples."""
    print("🚀 EmberAI Package Examples")
    print("=" * 50)
    
    # Step 1: Configure AI
    if not example_ai_configuration():
        print("\n❌ Cannot continue without valid API key")
        return
    
    # Step 2: General AI processing
    example_general_ai_processing()
    
    # Step 3: Filter configuration
    example_filter_configuration()
    
    # Step 4: Filtering examples
    await example_basic_filtering()
    await example_legal_filtering()
    await example_gender_enrichment()
    await example_combined_processing()
    
    print("\n✅ All examples completed!")
    print("\nNext steps:")
    print("1. Set your GEMINI_API_KEY environment variable")
    print("2. Modify the examples for your use case")
    print("3. Check the README.md for more advanced usage")

if __name__ == "__main__":
    asyncio.run(main()) 