"""
Tests for ModelConfig and ModelSource classes
"""
import os
import tempfile
import pytest
from opencode_pyrunner import (
    ModelConfig,
    ModelSource,
    ModelSourceType,
    load_env_file,
    parse_env_model_config,
)


class TestModelConfig:
    """Test cases for ModelConfig class"""

    def test_default_initialization(self):
        """Test ModelConfig with default values"""
        config = ModelConfig()
        assert config.baseurl == ""
        assert config.apikey == ""
        assert config.model == ""
        assert config.temperature is None
        assert config.max_tokens is None

    def test_full_initialization(self):
        """Test ModelConfig with all values"""
        config = ModelConfig(
            baseurl="https://api.openai.com",
            apikey="sk-test",
            model="gpt-4",
            temperature=0.7,
            max_tokens=4096,
        )
        assert config.baseurl == "https://api.openai.com"
        assert config.apikey == "sk-test"
        assert config.model == "gpt-4"
        assert config.temperature == 0.7
        assert config.max_tokens == 4096

    def test_to_dict(self):
        """Test conversion to dictionary"""
        config = ModelConfig(
            baseurl="https://api.openai.com",
            apikey="sk-test",
            model="gpt-4",
            temperature=0.7,
        )
        result = config.to_dict()
        assert result["baseurl"] == "https://api.openai.com"
        assert result["apikey"] == "sk-test"
        assert result["model"] == "gpt-4"
        assert result["temperature"] == 0.7
        assert "max_tokens" not in result

    def test_from_dict(self):
        """Test creation from dictionary"""
        data = {
            "baseurl": "https://api.openai.com",
            "apikey": "sk-test",
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 4096,
        }
        config = ModelConfig.from_dict(data)
        assert config.baseurl == "https://api.openai.com"
        assert config.apikey == "sk-test"
        assert config.model == "gpt-4"
        assert config.temperature == 0.7
        assert config.max_tokens == 4096


class TestModelSource:
    """Test cases for ModelSource class"""

    def test_default_initialization(self):
        """Test ModelSource with default values"""
        source = ModelSource()
        assert source.sources == {}

    def test_set_model(self):
        """Test setting model for a source type"""
        source = ModelSource()
        source.set_model(
            ModelSourceType.CODE,
            baseurl="https://api.openai.com",
            apikey="sk-test",
            model="gpt-4",
            temperature=0.3,
        )
        assert ModelSourceType.CODE in source.sources
        config = source.get_model(ModelSourceType.CODE)
        assert config is not None
        assert config.model == "gpt-4"
        assert config.temperature == 0.3

    def test_get_model_not_exists(self):
        """Test getting model that doesn't exist"""
        source = ModelSource()
        result = source.get_model(ModelSourceType.CODE)
        assert result is None

    def test_has_model(self):
        """Test checking if model exists"""
        source = ModelSource()
        assert not source.has_model(ModelSourceType.CODE)
        source.set_model(
            ModelSourceType.CODE,
            baseurl="https://api.openai.com",
            apikey="sk-test",
            model="gpt-4",
        )
        assert source.has_model(ModelSourceType.CODE)

    def test_to_dict(self):
        """Test conversion to dictionary"""
        source = ModelSource()
        source.set_model(
            ModelSourceType.CODE,
            baseurl="https://api.openai.com",
            apikey="sk-test",
            model="gpt-4",
        )
        result = source.to_dict()
        assert "code" in result
        assert result["code"]["model"] == "gpt-4"

    def test_from_dict(self):
        """Test creation from dictionary"""
        data = {
            "code": {
                "baseurl": "https://api.openai.com",
                "apikey": "sk-test",
                "model": "gpt-4",
            },
            "write": {
                "baseurl": "https://api.openai.com",
                "apikey": "sk-test",
                "model": "gpt-3.5-turbo",
            },
        }
        source = ModelSource.from_dict(data)
        assert source.has_model(ModelSourceType.CODE)
        assert source.has_model(ModelSourceType.WRITE)
        assert source.get_model(ModelSourceType.CODE).model == "gpt-4"
        assert source.get_model(ModelSourceType.WRITE).model == "gpt-3.5-turbo"


class TestEnvFileLoading:
    """Test cases for environment file loading"""

    def test_load_env_file_not_exists(self):
        """Test loading non-existent env file"""
        result = load_env_file("/nonexistent/path/.env")
        assert result == {}

    def test_load_env_file(self):
        """Test loading valid env file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("# Comment line\n")
            f.write("KEY1=value1\n")
            f.write('KEY2="value2"\n')
            f.write("KEY3='value3'\n")
            f.write("EMPTY=\n")
            temp_path = f.name

        try:
            result = load_env_file(temp_path)
            assert result["KEY1"] == "value1"
            assert result["KEY2"] == "value2"
            assert result["KEY3"] == "value3"
            # Empty values are kept in the result
            assert "EMPTY" in result
            assert result["EMPTY"] == ""
        finally:
            os.unlink(temp_path)

    def test_parse_env_model_config(self):
        """Test parsing model config from env vars"""
        env_vars = {
            "MODEL_CODE_BASEURL": "https://api.openai.com",
            "MODEL_CODE_APIKEY": "sk-code",
            "MODEL_CODE_MODEL": "gpt-4",
            "MODEL_CODE_TEMPERATURE": "0.3",
            "MODEL_CODE_MAX_TOKENS": "4096",
            "MODEL_WRITE_MODEL": "gpt-3.5-turbo",
        }
        source = parse_env_model_config(env_vars)
        assert source.has_model(ModelSourceType.CODE)
        assert source.has_model(ModelSourceType.WRITE)
        
        code_config = source.get_model(ModelSourceType.CODE)
        assert code_config.baseurl == "https://api.openai.com"
        assert code_config.apikey == "sk-code"
        assert code_config.model == "gpt-4"
        assert code_config.temperature == 0.3
        assert code_config.max_tokens == 4096
        
        write_config = source.get_model(ModelSourceType.WRITE)
        assert write_config.model == "gpt-3.5-turbo"

    def test_model_source_from_env_file(self):
        """Test ModelSource.from_env_file method"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("MODEL_CODE_MODEL=gpt-4\n")
            f.write("MODEL_WRITE_MODEL=gpt-3.5-turbo\n")
            temp_path = f.name

        try:
            source = ModelSource.from_env_file(temp_path)
            assert source.has_model(ModelSourceType.CODE)
            assert source.has_model(ModelSourceType.WRITE)
        finally:
            os.unlink(temp_path)
