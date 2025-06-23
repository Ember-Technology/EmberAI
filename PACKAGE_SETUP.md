# EmberAI Package Setup & Deployment Guide

This guide explains how to set up, deploy, and use the EmberAI package across multiple projects.

## Package Structure

```
EmberAI-package/
├── EmberAI/                    # Main package
│   ├── __init__.py            # Package entry point
│   ├── ai/                    # AI processing module
│   │   └── __init__.py        # Gemini API integration
│   └── filters.py             # Filtering and enrichment
├── examples/                  # Usage examples
│   └── basic_usage.py
├── tests/                     # Test suite
│   ├── __init__.py
│   └── test_ember_ai.py
├── setup.py                   # Package configuration
├── pyproject.toml             # Modern Python packaging
├── README.md                  # Documentation
├── LICENSE                    # MIT license
├── .gitignore                 # Git ignore patterns
└── PACKAGE_SETUP.md          # This file
```

## Installation Methods

### Method 1: GitHub Installation (Recommended)

Install directly from GitHub repository:

```bash
pip install https://github.com/Ember-Technology/EmberAI.git
```

### Method 2: Development Installation

For development and testing:

```bash
git clone https://github.com/Ember-Technology/EmberAI.git
cd EmberAI
pip install -e .
```

### Method 3: PyPI Installation (Future)

Once published to PyPI:

```bash
pip install EmberAI
```

## Quick Setup

### 1. Get Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the API key for use in your projects

### 2. Basic Usage

```python
import EmberAI

# Configure with your API key
EmberAI.ai.configure(GEMINI_API_KEY="your-api-key-here")

# Test AI processing
response = EmberAI.ai.process("Hello, how are you?")
print(response)

# Test filtering (async)
import asyncio

async def test_filtering():
    users = [{"username": "test", "full_name": "Test User"}]
    result = await EmberAI.filters.process(users, enrich_gender=True)
    print(result)

asyncio.run(test_filtering())
```

## Multi-Project Usage

### Project A Setup

```python
# project_a/main.py
import EmberAI

# Configure once per project
EmberAI.ai.configure(GEMINI_API_KEY="your-key")

# Use for EU filtering
async def filter_users(users):
    return await EmberAI.filters.process(
        users,
        block_eu_users=True,
        block_legal_users=True
    )
```

### Project B Setup

```python
# project_b/enrichment.py
import EmberAI

# Configure with different settings
EmberAI.ai.configure(
    GEMINI_API_KEY="your-key",
    model="gemini-1.5-pro",  # More powerful model
    temperature=0.2
)

# Use for gender enrichment
async def enrich_user_data(users):
    return await EmberAI.filters.process(
        users,
        enrich_gender=True,
        batch_size=50
    )
```

## Environment Variables

Set up environment variables for secure API key management:

### Linux/macOS

```bash
export GEMINI_API_KEY="your-api-key-here"
```

### Windows

```cmd
set GEMINI_API_KEY=your-api-key-here
```

### Python Code

```python
import os
import EmberAI

# Get API key from environment
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Please set GEMINI_API_KEY environment variable")

EmberAI.ai.configure(GEMINI_API_KEY=api_key)
```

## Configuration Management

### Global Configuration

```python
# config.py
import EmberAI

def setup_ember_ai():
    EmberAI.ai.configure(
        GEMINI_API_KEY="your-key",
        model="gemini-1.5-flash",
        temperature=0.1
    )

    EmberAI.filters.configure(
        batch_size=25,
        rate_limit_per_sec=50,
        max_concurrent=128
    )

# main.py
from config import setup_ember_ai
setup_ember_ai()
```

### Project-Specific Configuration

```python
# Different configuration per environment
import os
import EmberAI

if os.getenv("ENVIRONMENT") == "production":
    EmberAI.filters.configure(
        batch_size=50,
        rate_limit_per_sec=100,
        max_concurrent=256
    )
else:
    EmberAI.filters.configure(
        batch_size=10,
        rate_limit_per_sec=20,
        max_concurrent=32
    )
```

## Performance Optimization

### High-Volume Processing

```python
# For processing large datasets
EmberAI.filters.configure(
    batch_size=100,           # Larger batches
    rate_limit_per_sec=200,   # Higher rate limit
    max_concurrent=512,       # More concurrency
    preserve_order=False,     # Skip order preservation
    log_stats=False          # Reduce logging
)
```

### Memory-Efficient Processing

```python
async def process_large_dataset(users, chunk_size=1000):
    """Process large datasets in chunks to manage memory."""
    results = []

    for i in range(0, len(users), chunk_size):
        chunk = users[i:i + chunk_size]
        result = await EmberAI.filters.process(chunk, block_eu_users=True)
        results.extend(result['processed_users'])

        # Optional: clear memory between chunks
        import gc
        gc.collect()

    return results
```

## Error Handling

### Robust Error Handling

```python
import logging
import EmberAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def safe_processing(users):
    try:
        # Configure if not already done
        if not EmberAI.ai.is_configured():
            EmberAI.ai.configure(GEMINI_API_KEY="your-key")

        # Process with error handling
        result = await EmberAI.filters.process(
            users,
            block_eu_users=True,
            enrich_gender=True
        )

        logger.info(f"Successfully processed {result['final_count']} users")
        return result

    except RuntimeError as e:
        if "not configured" in str(e):
            logger.error("EmberAI not configured properly")
            raise
        else:
            logger.error(f"Processing error: {e}")
            raise

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        # Return empty result or re-raise based on needs
        return {
            'processed_users': [],
            'original_count': len(users),
            'final_count': 0,
            'filters_applied': [],
            'processing_stats': {'error': str(e)}
        }
```

## Testing

### Unit Tests

Run the test suite:

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest tests/

# Run with coverage
pip install pytest-cov
pytest tests/ --cov=EmberAI --cov-report=html
```

### Integration Testing

```python
# test_integration.py
import pytest
import EmberAI

@pytest.mark.asyncio
async def test_full_workflow():
    # Configure
    EmberAI.ai.configure(GEMINI_API_KEY="test-key")

    # Test data
    users = [
        {"username": "test", "full_name": "Test User", "location": "USA"}
    ]

    # Process
    result = await EmberAI.filters.process(users, enrich_gender=True)

    # Verify
    assert len(result['processed_users']) == 1
    assert 'gender' in result['processed_users'][0]
```

## Deployment Strategies

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install EmberAI
RUN pip install git+https://github.com/Ember-Technology/EmberAI.git

# Copy your application
COPY . .

# Set environment variable
ENV GEMINI_API_KEY=${GEMINI_API_KEY}

# Run your application
CMD ["python", "main.py"]
```

### Requirements File

```txt
# requirements.txt
EmberAI @ git+https://github.com/Ember-Technology/EmberAI.git
aiohttp>=3.8.0
fastapi>=0.68.0
```

### Cloud Functions

```python
# Google Cloud Function
import functions_framework
import EmberAI

# Configure once at module level
EmberAI.ai.configure(GEMINI_API_KEY="your-key")

@functions_framework.http
def process_users(request):
    import asyncio

    users = request.get_json()

    async def process():
        return await EmberAI.filters.process(
            users,
            block_eu_users=True
        )

    result = asyncio.run(process())
    return result
```

## Version Management

### Package Updates

```bash
# Update to latest version
pip install --upgrade git+https://github.com/Ember-Technology/EmberAI.git

# Install specific version (when tagged)
pip install git+https://github.com/Ember-Technology/EmberAI@v1.1.0
```

### Version Checking

```python
import EmberAI
print(f"EmberAI version: {EmberAI.__version__}")
```

## Common Issues & Solutions

### Issue 1: API Key Not Set

```
RuntimeError: EmberAI not configured
```

**Solution:**

```python
import os
EmberAI.ai.configure(GEMINI_API_KEY=os.getenv("GEMINI_API_KEY"))
```

### Issue 2: Rate Limit Exceeded

```
Rate limit exceeded
```

**Solution:**

```python
EmberAI.filters.configure(
    rate_limit_per_sec=10,
    max_concurrent=32
)
```

### Issue 3: Memory Issues

```
MemoryError: Out of memory
```

**Solution:**

```python
# Use smaller batch sizes
EmberAI.filters.configure(batch_size=5)

# Or process in chunks
async def process_in_chunks(users, chunk_size=100):
    for chunk in [users[i:i+chunk_size] for i in range(0, len(users), chunk_size)]:
        result = await EmberAI.filters.process(chunk)
        yield result
```

## Support & Maintenance

### Getting Help

1. **Documentation**: Check README.md for comprehensive usage guide
2. **Examples**: See examples/ directory for working code
3. **Issues**: Report bugs on GitHub Issues
4. **Testing**: Run tests locally before deployment

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### Package Maintenance

- Regular dependency updates
- Security patches
- Performance improvements
- New feature additions

## License

This package is released under the MIT License. See LICENSE file for details.
