import pytest
from unittest.mock import patch, MagicMock
import sys
import os

sys.modules['vertexai'] = MagicMock()
sys.modules['vertexai.generative_models'] = MagicMock()
sys.modules['google.cloud.aiplatform_v1beta1.types'] = MagicMock()

@patch.dict(os.environ, {'VERTEX_PROJECT_ID': 'test-project', 'VERTEX_LOCATION': 'us-east1'})
def test_plugin_is_installed():
    """Test that the plugin can be imported and is available."""
    try:
        import llm_vertex
        assert hasattr(llm_vertex, 'register_models')
        assert hasattr(llm_vertex, 'Vertex')
    except ImportError:
        pytest.fail("Failed to import llm_vertex module")


@patch.dict(os.environ, {'VERTEX_PROJECT_ID': 'test-project', 'VERTEX_LOCATION': 'us-east1'})
def test_supported_models_list():
    """Test that key models are available in the register_models function."""
    import llm_vertex

    # Test for a few key models that should always be present
    # This won't break when new models are added or preview models are removed
    key_models = [
        'gemini-1.5-pro',
        'gemini-1.5-flash',
    ]

    registered_models = []
    def mock_register(model):
        registered_models.append(model)

    llm_vertex.register_models(mock_register)

    # Check that we have at least some models registered
    assert len(registered_models) > 0, "No models were registered"
    registered_model_names = [model.model_name for model in registered_models]
    for key_model in key_models:
        assert key_model in registered_model_names, f"Key model {key_model} not found in registered models"

    for model in registered_models:
        assert model.model_id.startswith('vertex-'), f"Model {model.model_id} doesn't have vertex- prefix"


@patch.dict(os.environ, {'VERTEX_PROJECT_ID': 'test-project', 'VERTEX_LOCATION': 'us-east1'})
def test_vertex_model_initialization():
    """Test that we can create a Vertex model instance without errors."""
    import llm_vertex

    # Test that we can create a Vertex model instance
    model = llm_vertex.Vertex("vertex-gemini-1.0-pro")
    assert model.model_id == "vertex-gemini-1.0-pro"
    assert model.model_name == "gemini-1.0-pro"
    assert model.can_stream is True


@patch.dict(os.environ, {'VERTEX_PROJECT_ID': 'test-project', 'VERTEX_LOCATION': 'us-east1'})
def test_vertex_model_options():
    """Test that the Vertex model has the expected options."""
    import llm_vertex

    model = llm_vertex.Vertex("vertex-gemini-1.0-pro")

    # Check that the Options class exists and has expected fields
    assert hasattr(model, 'Options')
    options_class = model.Options

    # Check that the options have the expected fields
    # These should be defined in the Options class
    option_fields = ['max_output_tokens', 'temperature', 'top_p', 'top_k']

    # Create an instance to check the fields exist
    options = options_class()
    for field in option_fields:
        assert hasattr(options, field), f"Option field {field} not found"


def test_model_name_extraction():
    """Test that model names are correctly extracted from model IDs."""
    import llm_vertex

    test_cases = [
        ("vertex-gemini-1.0-pro", "gemini-1.0-pro"),
        ("vertex-gemini-1.5-flash", "gemini-1.5-flash"),
        ("vertex-gemini-2.0-flash-001", "gemini-2.0-flash-001"),
    ]

    for model_id, expected_name in test_cases:
        with patch.dict(os.environ, {'VERTEX_PROJECT_ID': 'test-project', 'VERTEX_LOCATION': 'us-east1'}):
            model = llm_vertex.Vertex(model_id)
            assert model.model_name == expected_name, f"Expected {expected_name}, got {model.model_name}"
