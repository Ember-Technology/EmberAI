"""
Tests for EmberAI package.

This module contains tests for both the AI processing and filtering functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import EmberAI
from EmberAI.filters import FilterConfig, FilterProcessor

# Test data
SAMPLE_USERS = [
    {
        "username": "john_doe",
        "full_name": "John Doe",
        "description": "Software engineer",
        "location": "San Francisco, CA"
    },
    {
        "username": "jane_smith",
        "full_name": "Jane Smith",
        "description": "Attorney at law firm",
        "location": "Berlin, Germany",
        "education": "Harvard Law School"
    }
]

class TestAIModule:
    """Test cases for EmberAI.ai module."""
    
    def test_ai_not_configured_initially(self):
        """Test that AI is not configured initially."""
        # Reset configuration
        EmberAI.ai._processor._configured = False
        assert not EmberAI.ai.is_configured()
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_ai_configuration(self, mock_model, mock_configure):
        """Test AI configuration."""
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance
        
        # Configure AI
        EmberAI.ai.configure(GEMINI_API_KEY="test-key")
        
        # Verify configuration
        assert EmberAI.ai.is_configured()
        mock_configure.assert_called_once_with(api_key="test-key")
        mock_model.assert_called_once()
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_ai_processing(self, mock_model, mock_configure):
        """Test AI processing."""
        # Setup mocks
        mock_model_instance = Mock()
        mock_response = Mock()
        mock_response.text = "Test response"
        mock_response.candidates = [Mock()]
        mock_model_instance.generate_content.return_value = mock_response
        mock_model.return_value = mock_model_instance
        
        # Configure and process
        EmberAI.ai.configure(GEMINI_API_KEY="test-key")
        result = EmberAI.ai.process("Test prompt")
        
        # Verify
        assert result == "Test response"
        mock_model_instance.generate_content.assert_called_once_with("Test prompt")
    
    def test_ai_processing_not_configured(self):
        """Test AI processing when not configured."""
        # Reset configuration
        EmberAI.ai._processor._configured = False
        
        with pytest.raises(RuntimeError, match="EmberAI not configured"):
            EmberAI.ai.process("Test prompt")
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_get_model_info(self, mock_model, mock_configure):
        """Test getting model information."""
        mock_model_instance = Mock()
        mock_model_instance.model_name = "gemini-1.5-flash"
        mock_model.return_value = mock_model_instance
        
        # Configure AI
        EmberAI.ai.configure(GEMINI_API_KEY="test-key")
        
        # Get model info
        info = EmberAI.ai.get_model_info()
        
        assert info["configured"] is True
        assert info["api_key_set"] is True
        assert info["model"] == "gemini-1.5-flash"

class TestFilterConfig:
    """Test cases for FilterConfig."""
    
    def test_default_config(self):
        """Test default filter configuration."""
        config = FilterConfig()
        
        assert config.block_eu_users is False
        assert config.block_legal_users is False
        assert config.enrich_gender is False
        assert config.batch_size == 25
        assert config.rate_limit_per_sec == 50
        assert config.max_concurrent == 128
        assert config.preserve_order is True
        assert config.log_stats is True
    
    def test_custom_config(self):
        """Test custom filter configuration."""
        config = FilterConfig(
            block_eu_users=True,
            batch_size=10,
            rate_limit_per_sec=20
        )
        
        assert config.block_eu_users is True
        assert config.batch_size == 10
        assert config.rate_limit_per_sec == 20
        # Other values should be defaults
        assert config.block_legal_users is False
        assert config.max_concurrent == 128

class TestFilterProcessor:
    """Test cases for FilterProcessor."""
    
    def test_processor_initialization(self):
        """Test filter processor initialization."""
        config = FilterConfig(batch_size=10)
        processor = FilterProcessor(config)
        
        assert processor.config.batch_size == 10
        assert processor._limiter is not None
        assert processor._semaphore is not None
    
    @pytest.mark.asyncio
    async def test_process_empty_users(self):
        """Test processing empty user list."""
        config = FilterConfig()
        processor = FilterProcessor(config)
        
        result = await processor.process_filters([])
        
        assert result['processed_users'] == []
        assert result['original_count'] == 0
        assert result['final_count'] == 0
        assert result['filters_applied'] == []
    
    @pytest.mark.asyncio
    @patch('EmberAI.ai.get_instance')
    async def test_process_not_configured(self, mock_get_instance):
        """Test processing when AI not configured."""
        mock_ai = Mock()
        mock_ai.is_configured.return_value = False
        mock_get_instance.return_value = mock_ai
        
        config = FilterConfig()
        processor = FilterProcessor(config)
        
        with pytest.raises(RuntimeError, match="EmberAI not configured"):
            await processor.process_filters(SAMPLE_USERS)
    
    def test_build_eu_prompt(self):
        """Test EU prompt building."""
        config = FilterConfig()
        processor = FilterProcessor(config)
        
        prompt = processor._build_eu_prompt(SAMPLE_USERS[:1])
        
        assert "EU residents" in prompt
        assert "John Doe" in prompt
        assert "San Francisco" in prompt
        assert "<classification>" in prompt
    
    def test_build_legal_prompt(self):
        """Test legal prompt building."""
        config = FilterConfig()
        processor = FilterProcessor(config)
        
        prompt = processor._build_legal_prompt(SAMPLE_USERS[:1])
        
        assert "legal professionals" in prompt
        assert "John Doe" in prompt
        assert "Software engineer" in prompt
        assert "<classification>" in prompt
    
    def test_build_gender_prompt(self):
        """Test gender prompt building."""
        config = FilterConfig()
        processor = FilterProcessor(config)
        
        prompt = processor._build_gender_prompt(SAMPLE_USERS[:1])
        
        assert "gender" in prompt
        assert "John Doe" in prompt
        assert "john_doe" in prompt
        assert "<gender>" in prompt
    
    def test_parse_eu_response(self):
        """Test parsing EU classification response."""
        config = FilterConfig()
        processor = FilterProcessor(config)
        
        response = """<classification>EU</classification>
<classification>NOT_EU</classification>"""
        
        result = processor._parse_eu_response(response, 2)
        
        assert result == [True, False]
    
    def test_parse_legal_response(self):
        """Test parsing legal classification response."""
        config = FilterConfig()
        processor = FilterProcessor(config)
        
        response = """<classification>LEGAL</classification>
<classification>NOT_LEGAL</classification>"""
        
        result = processor._parse_legal_response(response, 2)
        
        assert result == [True, False]
    
    def test_parse_gender_response(self):
        """Test parsing gender classification response."""
        config = FilterConfig()
        processor = FilterProcessor(config)
        
        response = """<gender>male</gender>
<gender>female</gender>
<gender>unknown</gender>"""
        
        result = processor._parse_gender_response(response, 3)
        
        assert result == ["male", "female", "unknown"]

class TestFiltersModule:
    """Test cases for EmberAI.filters module."""
    
    def test_configure_filters(self):
        """Test filter configuration."""
        EmberAI.filters.configure(
            batch_size=10,
            rate_limit_per_sec=20
        )
        
        config = EmberAI.filters.get_config()
        assert config.batch_size == 10
        assert config.rate_limit_per_sec == 20
    
    @pytest.mark.asyncio
    @patch('EmberAI.ai.get_instance')
    async def test_process_function(self, mock_get_instance):
        """Test the process function."""
        # Mock AI instance
        mock_ai = Mock()
        mock_ai.is_configured.return_value = True
        mock_ai.process.return_value = "<classification>NOT_EU</classification>"
        mock_get_instance.return_value = mock_ai
        
        # Test process function
        result = await EmberAI.filters.process(
            SAMPLE_USERS[:1],
            block_eu_users=True
        )
        
        assert 'processed_users' in result
        assert 'original_count' in result
        assert 'final_count' in result
        assert 'filters_applied' in result
        assert result['original_count'] == 1

class TestIntegration:
    """Integration tests for the full package."""
    
    @pytest.mark.asyncio
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    async def test_full_workflow(self, mock_model, mock_configure):
        """Test complete workflow from configuration to processing."""
        # Setup mocks
        mock_model_instance = Mock()
        mock_response = Mock()
        mock_response.text = "<classification>NOT_EU</classification>"
        mock_response.candidates = [Mock()]
        mock_model_instance.generate_content.return_value = mock_response
        mock_model.return_value = mock_model_instance
        
        # Configure AI
        EmberAI.ai.configure(GEMINI_API_KEY="test-key")
        assert EmberAI.ai.is_configured()
        
        # Configure filters
        EmberAI.filters.configure(batch_size=5)
        
        # Process users
        result = await EmberAI.filters.process(
            SAMPLE_USERS[:1],
            block_eu_users=True
        )
        
        # Verify results
        assert result['original_count'] == 1
        assert result['final_count'] == 1  # Should not be blocked
        assert 'block_eu_users' in result['filters_applied']
    
    def test_package_imports(self):
        """Test that package imports work correctly."""
        import EmberAI
        import EmberAI.ai
        import EmberAI.filters
        
        # Test that modules are available
        assert hasattr(EmberAI, 'ai')
        assert hasattr(EmberAI, 'filters')
        
        # Test that functions are available
        assert hasattr(EmberAI.ai, 'configure')
        assert hasattr(EmberAI.ai, 'process')
        assert hasattr(EmberAI.ai, 'is_configured')
        
        assert hasattr(EmberAI.filters, 'configure')
        assert hasattr(EmberAI.filters, 'process')

if __name__ == "__main__":
    pytest.main([__file__]) 