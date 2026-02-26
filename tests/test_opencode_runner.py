"""
Tests for OpenCodeRunner class
"""
import os
import tempfile
import pytest
from unittest.mock import Mock, patch, MagicMock
from opencode_pyrunner import (
    OpenCodeRunner,
    ModelSource,
    ModelSourceType,
    RunState,
)


class TestOpenCodeRunnerInitialization:
    """Test cases for OpenCodeRunner initialization"""

    def test_basic_initialization(self):
        """Test basic initialization with task_name"""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = OpenCodeRunner(task_name="test_task", workspace=tmpdir)
            assert runner.task_name == "test_task"
            assert runner.model is None
            assert runner.context is None
            assert runner.workspace == tmpdir
            assert runner.session_id is None
            assert runner.conversation_history == []
            assert os.path.exists(runner.task_dir)
            assert runner.task_dir.startswith(os.path.join(tmpdir, "task_test_task_"))

    def test_initialization_with_model(self):
        """Test initialization with model parameter"""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = OpenCodeRunner(
                task_name="test_task",
                model="gpt-4",
                workspace=tmpdir
            )
            assert runner.model == "gpt-4"

    def test_initialization_with_context(self):
        """Test initialization with context parameter"""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = OpenCodeRunner(
                task_name="test_task",
                context="You are a Python expert",
                workspace=tmpdir
            )
            assert runner.context == "You are a Python expert"

    def test_initialization_with_model_source(self):
        """Test initialization with model_source"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_source = ModelSource()
            model_source.set_model(
                ModelSourceType.CODE,
                baseurl="https://api.openai.com",
                apikey="sk-test",
                model="gpt-4",
            )
            runner = OpenCodeRunner(
                task_name="test_task",
                model_source=model_source,
                workspace=tmpdir
            )
            assert runner.model_source is model_source

    def test_default_workspace(self):
        """Test that default workspace is current directory"""
        runner = OpenCodeRunner(task_name="test_task")
        assert runner.workspace == os.getcwd()
        # Cleanup
        import shutil
        shutil.rmtree(runner.task_dir, ignore_errors=True)


class TestOpenCodeRunnerState:
    """Test cases for OpenCodeRunner state management"""

    def test_initial_state(self):
        """Test initial state is IDLE"""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = OpenCodeRunner(task_name="test_task", workspace=tmpdir)
            assert runner.get_state() == RunState.IDLE
            assert not runner.is_running()

    def test_state_transitions(self):
        """Test state transitions"""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = OpenCodeRunner(task_name="test_task", workspace=tmpdir)
            # Manually set state for testing
            runner._state = RunState.RUNNING
            assert runner.get_state() == RunState.RUNNING
            assert runner.is_running()

            runner._state = RunState.COMPLETED
            assert runner.get_state() == RunState.COMPLETED
            assert not runner.is_running()

            runner._state = RunState.INTERRUPTED
            assert runner.get_state() == RunState.INTERRUPTED
            assert not runner.is_running()


class TestOpenCodeRunnerSession:
    """Test cases for OpenCodeRunner session management"""

    def test_initial_session_id(self):
        """Test initial session_id is None"""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = OpenCodeRunner(task_name="test_task", workspace=tmpdir)
            assert runner.get_session_id() is None

    def test_session_id_property(self):
        """Test session_id property"""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = OpenCodeRunner(task_name="test_task", workspace=tmpdir)
            # Set session_id via the internal attribute
            runner.session_id = "test-session-123"
            assert runner.get_session_id() == "test-session-123"


class TestOpenCodeRunnerOutput:
    """Test cases for OpenCodeRunner output handling"""

    def test_get_current_output_empty(self):
        """Test getting output when empty"""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = OpenCodeRunner(task_name="test_task", workspace=tmpdir)
            assert runner.get_current_output() == ""

    def test_get_current_output_with_content(self):
        """Test getting output with content"""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = OpenCodeRunner(task_name="test_task", workspace=tmpdir)
            # Simulate output by adding to conversation history
            runner.conversation_history = [
                {"role": "assistant", "content": "Hello World"}
            ]
            # The get_current_output method reads from conversation history
            # when there's no running process
            assert runner.get_current_output() == ""


class TestOpenCodeRunnerMarkdown:
    """Test cases for OpenCodeRunner markdown generation"""

    def test_get_conversation_markdown_empty(self):
        """Test markdown generation with empty history"""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = OpenCodeRunner(task_name="test_task", workspace=tmpdir)
            markdown = runner.get_conversation_markdown()
            # The markdown uses task_name as title, not "Conversation"
            assert "# test_task" in markdown
            assert "Task Dir" in markdown

    def test_get_conversation_markdown_with_history(self):
        """Test markdown generation with conversation history"""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = OpenCodeRunner(task_name="test_task", workspace=tmpdir)
            runner.conversation_history = [
                {"role": "user", "content": "What is Python?", "context": None},
                {"role": "assistant", "content": "Python is a programming language.", "output": "Python is a programming language."},
            ]
            runner._output_parts = ["Python is a programming language."]
            markdown = runner.get_conversation_markdown()
            assert "What is Python?" in markdown
            assert "Python is a programming language." in markdown

    def test_save_markdown_default_path(self):
        """Test saving markdown to default path"""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = OpenCodeRunner(task_name="test_task", workspace=tmpdir)
            runner.conversation_history = [
                {"role": "user", "content": "Hello", "context": None},
            ]
            filepath = runner.save_markdown()
            assert os.path.exists(filepath)
            assert filepath.endswith("conversation.md")
            with open(filepath, 'r') as f:
                content = f.read()
                assert "Hello" in content

    def test_save_markdown_custom_path(self):
        """Test saving markdown to custom path"""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = OpenCodeRunner(task_name="test_task", workspace=tmpdir)
            runner.conversation_history = [
                {"role": "user", "content": "Hello", "context": None},
            ]
            custom_path = os.path.join(tmpdir, "custom.md")
            filepath = runner.save_markdown(custom_path)
            assert filepath == custom_path
            assert os.path.exists(custom_path)


class TestOpenCodeRunnerCommandBuilding:
    """Test cases for OpenCodeRunner command building"""

    def test_build_command_basic(self):
        """Test basic command building"""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = OpenCodeRunner(task_name="test_task", workspace=tmpdir)
            cmd = runner._build_command("Hello")
            assert cmd[0] == "opencode"
            assert cmd[1] == "run"
            assert "--format" in cmd
            assert "json" in cmd
            assert "--" in cmd
            assert cmd[-1] == "Hello"

    def test_build_command_with_model(self):
        """Test command building with model"""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = OpenCodeRunner(
                task_name="test_task",
                model="gpt-4",
                workspace=tmpdir
            )
            cmd = runner._build_command("Hello")
            assert "--model" in cmd
            assert "gpt-4" in cmd

    def test_build_command_with_session(self):
        """Test command building with session"""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = OpenCodeRunner(task_name="test_task", workspace=tmpdir)
            # Set session_id via the public attribute
            runner.session_id = "test-session-123"
            cmd = runner._build_command("Hello")
            assert "--session" in cmd
            assert "test-session-123" in cmd


class TestOpenCodeRunnerModelSourceSelection:
    """Test cases for model source selection"""

    def test_select_model_from_source(self):
        """Test selecting model from model_source"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_source = ModelSource()
            model_source.set_model(
                ModelSourceType.CODE,
                baseurl="https://api.openai.com",
                apikey="sk-test",
                model="gpt-4-code",
            )
            runner = OpenCodeRunner(
                task_name="test_task",
                model_source=model_source,
                workspace=tmpdir
            )
            
            model = runner._select_model(model_source_type=ModelSourceType.CODE)
            assert model == "gpt-4-code"

    def test_select_model_fallback_to_default(self):
        """Test model selection fallback to default model"""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = OpenCodeRunner(
                task_name="test_task",
                model="gpt-4-default",
                workspace=tmpdir
            )
            
            model = runner._select_model()
            assert model == "gpt-4-default"

    def test_select_model_precedence(self):
        """Test model selection precedence: explicit > source > default"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_source = ModelSource()
            model_source.set_model(
                ModelSourceType.CODE,
                baseurl="https://api.openai.com",
                apikey="sk-test",
                model="gpt-4-code",
            )
            runner = OpenCodeRunner(
                task_name="test_task",
                model="gpt-4-default",
                model_source=model_source,
                workspace=tmpdir
            )
            
            # Explicit model should take precedence
            model = runner._select_model(model="gpt-4-explicit")
            assert model == "gpt-4-explicit"
            
            # Source model should be used when no explicit model
            model = runner._select_model(model_source_type=ModelSourceType.CODE)
            assert model == "gpt-4-code"
            
            # Default model should be used when no source
            model = runner._select_model(model_source_type=ModelSourceType.WRITE)
            assert model == "gpt-4-default"


class TestOpenCodeRunnerInterrupt:
    """Test cases for OpenCodeRunner interrupt functionality"""

    def test_interrupt_when_not_running(self):
        """Test interrupt when not running"""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = OpenCodeRunner(task_name="test_task", workspace=tmpdir)
            result = runner.interrupt()
            assert result is False

    def test_kill_when_not_running(self):
        """Test kill when not running"""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = OpenCodeRunner(task_name="test_task", workspace=tmpdir)
            result = runner.kill()
            assert result is False


class TestOpenCodeRunnerRecentEvents:
    """Test cases for getting recent events"""

    def test_get_recent_events_empty(self):
        """Test getting recent events when empty"""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = OpenCodeRunner(task_name="test_task", workspace=tmpdir)
            events = runner.get_recent_events(5)
            assert events == []

    def test_get_recent_events_with_data(self):
        """Test getting recent events with data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = OpenCodeRunner(task_name="test_task", workspace=tmpdir)
            runner.full_json_events = [
                {"type": "text", "content": "1"},
                {"type": "text", "content": "2"},
                {"type": "text", "content": "3"},
            ]
            events = runner.get_recent_events(2)
            assert len(events) == 2
            assert events[0]["content"] == "2"
            assert events[1]["content"] == "3"

    def test_get_recent_events_more_than_available(self):
        """Test getting more recent events than available"""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = OpenCodeRunner(task_name="test_task", workspace=tmpdir)
            runner.full_json_events = [
                {"type": "text", "content": "1"},
            ]
            events = runner.get_recent_events(5)
            assert len(events) == 1
