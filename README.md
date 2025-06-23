# EmberAI

Comprehensive AI processing system with filtering and enrichment capabilities powered by Google's Gemini API.

## Features

- **General AI Processing**: Direct access to Gemini API with optimized configuration
- **EU Location Detection**: AI-powered identification of EU residents with optional name-based filtering
- **Legal Professional Detection**: Identify and filter lawyers, attorneys, and legal professionals
- **Gender Enrichment**: Add gender information to user profiles using name analysis
- **High Performance**: Async processing with configurable rate limiting and concurrency
- **Flexible Configuration**: Toggle filters on/off with custom batch sizes and performance tuning

## Installation

### From GitHub (Recommended)

```bash
pip install git+https://github.com/yourusername/EmberAI.git
```

### Development Installation

```bash
git clone https://github.com/yourusername/EmberAI.git
cd EmberAI
pip install -e .
```

## Quick Start

### 1. Configuration

```python
import EmberAI

# Configure AI processor with your Gemini API key
EmberAI.ai.configure(GEMINI_API_KEY="your-gemini-api-key-here")

# Optional: Configure filters with custom settings
EmberAI.filters.configure(
    batch_size=25,
    rate_limit_per_sec=50,
    max_concurrent=128
)
```

### 2. General AI Processing

```python
# Direct AI processing
response = EmberAI.ai.process("What is the capital of France?")
print(response)  # "The capital of France is Paris."

# Custom generation parameters
response = EmberAI.ai.process(
    "Write a short poem about coding",
    temperature=0.7,
    max_output_tokens=200
)
```

### 3. AI-Powered Filtering

```python
import asyncio

# Sample user data
users = [
    {
        "username": "john_attorney",
        "full_name": "John Smith",
        "description": "Lawyer at BigLaw Firm",
        "location": "Berlin, Germany",
        "education": "Harvard Law School"
    },
    {
        "username": "sarah_dev",
        "full_name": "Sarah Johnson",
        "description": "Software engineer at Google",
        "location": "San Francisco, CA"
    }
]

# Process with multiple filters
async def process_users():
    result = await EmberAI.filters.process(
        users,
        block_eu_users=True,      # Remove EU residents
        block_legal_users=True,   # Remove legal professionals
        enrich_gender=True        # Add gender information
    )

    print(f"Original: {result['original_count']} users")
    print(f"Final: {result['final_count']} users")
    print(f"Applied filters: {result['filters_applied']}")

    for user in result['processed_users']:
        print(f"User: {user['full_name']}, Gender: {user.get('gender', 'N/A')}")

# Run the async function
asyncio.run(process_users())
```

## Configuration Options

| Option               | Type | Default | Description                                 |
| -------------------- | ---- | ------- | ------------------------------------------- |
| `block_eu_users`     | bool | False   | Block users detected as EU residents        |
| `block_legal_users`  | bool | False   | Block users detected as legal professionals |
| `enrich_gender`      | bool | False   | Add gender information to user records      |
| `block_unique_names` | bool | False   | Enhanced EU blocking using name patterns    |
| `batch_size`         | int  | 25      | Number of users processed per batch         |
| `rate_limit_per_sec` | int  | 50      | API calls per second limit                  |
| `max_concurrent`     | int  | 128     | Maximum concurrent requests                 |
| `preserve_order`     | bool | True    | Maintain original user order                |
| `log_stats`          | bool | True    | Enable detailed logging                     |

## Expected Data Format

### Input User Format

```python
user = {
    # Core identification
    "username": "john_doe",
    "full_name": "John Doe",              # or "name"

    # Profile information
    "description": "Software engineer passionate about AI",
    "location": "San Francisco, CA",

    # Professional details (for legal detection)
    "education": "Stanford Computer Science",
    "employer": "TechCorp Inc",

    # Any additional fields will be preserved
    "email": "john@example.com",
    "age": 28
}
```

### Output Format

```python
{
    "processed_users": [
        {
            "username": "sarah_dev",
            "full_name": "Sarah Johnson",
            "description": "Software engineer",
            "gender": "female",  # Added by enrich_gender
            # ... other original fields preserved
        }
    ],
    "original_count": 100,
    "final_count": 75,
    "filters_applied": ["block_eu_users", "enrich_gender"],
    "processing_stats": {
        "total_time_seconds": 2.5,
        "processing_rate": 40.0
    }
}
```

## Advanced Usage

### Custom AI Configuration

```python
# Advanced AI configuration
EmberAI.ai.configure(
    GEMINI_API_KEY="your-key",
    model="gemini-1.5-pro",  # Use more powerful model
    temperature=0.2,
    top_p=0.9,
    max_output_tokens=4096
)
```

### Per-Request Filter Configuration

```python
# Override configuration for specific requests
result = await EmberAI.filters.process(
    users,
    block_eu_users=True,
    batch_size=50,          # Override default batch size
    rate_limit_per_sec=25,  # Override rate limit
    log_stats=False         # Disable logging for this request
)
```

### Reusable Filter Instances

```python
from EmberAI.filters import FilterProcessor, FilterConfig

# Create custom configuration
config = FilterConfig(
    block_eu_users=True,
    enrich_gender=True,
    batch_size=50,
    rate_limit_per_sec=30
)

# Create reusable processor
processor = FilterProcessor(config)

# Process multiple batches with same configuration
result1 = await processor.process_filters(batch1)
result2 = await processor.process_filters(batch2)
```

## Performance Optimization

### High-Throughput Settings

```python
# For maximum performance
EmberAI.filters.configure(
    batch_size=50,           # Larger batches
    rate_limit_per_sec=100,  # Higher rate limit
    max_concurrent=256,      # More concurrent requests
    preserve_order=False,    # Skip order preservation
    log_stats=False         # Reduce logging overhead
)
```

### Memory-Efficient Processing

```python
# Process large datasets in chunks
async def process_large_dataset(all_users, chunk_size=1000):
    all_results = []

    for i in range(0, len(all_users), chunk_size):
        chunk = all_users[i:i + chunk_size]
        result = await EmberAI.filters.process(chunk, block_eu_users=True)
        all_results.extend(result['processed_users'])

    return all_results
```

## Error Handling

```python
import EmberAI

try:
    # Configure AI
    EmberAI.ai.configure(GEMINI_API_KEY="your-key")

    # Process users
    result = await EmberAI.filters.process(users, block_eu_users=True)

except RuntimeError as e:
    if "not configured" in str(e):
        print("Please configure EmberAI with your API key first")
    else:
        print(f"Processing error: {e}")

except Exception as e:
    print(f"Unexpected error: {e}")
```

## Troubleshooting

### Common Issues

1. **"EmberAI not configured" Error**

   ```python
   # Make sure to configure before using
   EmberAI.ai.configure(GEMINI_API_KEY="your-key")
   ```

2. **Rate Limit Errors**

   ```python
   # Reduce rate limits
   EmberAI.filters.configure(rate_limit_per_sec=10, max_concurrent=32)
   ```

3. **Memory Issues with Large Datasets**
   ```python
   # Use smaller batch sizes
   EmberAI.filters.configure(batch_size=10)
   ```

## API Reference

### EmberAI.ai

- `configure(GEMINI_API_KEY, model="gemini-1.5-flash", **kwargs)` - Configure AI processor
- `process(prompt, **kwargs)` - Process a text prompt
- `is_configured()` - Check if configured
- `get_model_info()` - Get model information

### EmberAI.filters

- `configure(**kwargs)` - Configure filter settings
- `process(users, **kwargs)` - Process users with filters
- `FilterConfig` - Configuration dataclass
- `FilterProcessor` - Advanced processor class

## Requirements

- Python 3.8+
- google-generativeai>=0.3.0
- aiolimiter>=1.0.0

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions:

- GitHub Issues: [https://github.com/yourusername/EmberAI/issues](https://github.com/yourusername/EmberAI/issues)
- Email: your.email@example.com
