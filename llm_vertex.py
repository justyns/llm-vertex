import llm
import vertexai
import os
import json
import time
from pathlib import Path
from typing import Optional, List
# from google.cloud.aiplatform_v1beta1.types import Content, Part
from vertexai.generative_models import GenerativeModel, Part, ChatSession, Content, GenerationConfig

# Cache for available models to avoid repeated API calls
_cached_models: Optional[List[str]] = None

# Fallback list of known Gemini models
# Source: https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models
FALLBACK_MODELS = [
    'gemini-2.5-flash',
    'gemini-2.5-pro',
    'gemini-2.0-flash-lite',
    'gemini-2.0-flash',
    'gemini-1.5-pro',
    'gemini-1.5-flash',
]


def get_cache_file_path() -> Path:
    """
    Get the path to the cache file for storing model lists.
    Uses ~/.cache/llm-vertex/models.json on Unix-like systems.
    """
    cache_dir = Path.home() / '.cache' / 'llm-vertex'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / 'models.json'


def load_models_from_cache() -> Optional[List[str]]:
    """
    Load models from the cache file if it exists and is not expired.
    Returns None if cache doesn't exist, is expired, or cannot be read.
    Cache TTL defaults to 24 hours, configurable via VERTEX_CACHE_TTL environment variable.
    """
    try:
        cache_file = get_cache_file_path()
        if not cache_file.exists():
            return None

        with open(cache_file, 'r') as f:
            cache_data = json.load(f)

        # Check if cache is expired (default 24 hours)
        cached_time = cache_data.get('timestamp', 0)
        try:
            cache_ttl = int(os.getenv('VERTEX_CACHE_TTL', '86400'))
        except ValueError:
            cache_ttl = 86400

        if time.time() - cached_time > cache_ttl:
            return None

        return cache_data.get('models') or None

    except Exception as e:
        # If any error occurs reading cache, just return None
        print(f"Warning: Could not read cache file: {e}")
        return None


def save_models_to_cache(models: List[str]) -> None:
    """
    Save models to the cache file with current timestamp.
    """
    try:
        cache_file = get_cache_file_path()
        cache_data = {
            'timestamp': time.time(),
            'models': models
        }

        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)

    except Exception as e:
        # If we can't write cache, just log and continue
        print(f"Warning: Could not write cache file: {e}")


def _cache_and_return(models: List[str]) -> List[str]:
    """
    Helper function to cache models (both in-memory and file) and return them.
    """
    global _cached_models
    _cached_models = models
    save_models_to_cache(models)
    return models


def get_available_models() -> List[str]:
    """
    Fetch available Gemini models from Vertex AI.
    Falls back to a hardcoded list if the API call fails.
    Results are cached in memory and on disk to avoid repeated API calls.
    """
    global _cached_models

    # Return in-memory cached models if available
    if _cached_models is not None:
        return _cached_models

    # Try to load from file cache
    cached_from_file = load_models_from_cache()
    if cached_from_file:
        _cached_models = cached_from_file
        return cached_from_file

    # Check if dynamic model fetching is disabled
    if os.getenv('VERTEX_DISABLE_DYNAMIC_MODELS', '').lower() in ('true', '1', 'yes'):
        return _cache_and_return(FALLBACK_MODELS)

    try:
        # Try to get models dynamically from Vertex AI
        from google.cloud import aiplatform

        project_id = os.getenv('VERTEX_PROJECT_ID')
        location = os.getenv('VERTEX_LOCATION', 'us-central1')

        if not project_id:
            return _cache_and_return(FALLBACK_MODELS)

        # Initialize aiplatform and get publisher models
        aiplatform.init(project=project_id, location=location)
        from google.cloud.aiplatform_v1beta1 import ModelGardenServiceClient

        client = ModelGardenServiceClient()

        # List models from Model Garden
        models = []
        try:
            page_result = client.list_publisher_models(parent="publishers/google")

            for model in page_result:
                # Filter for Gemini models only
                if model.name and 'gemini' in model.name.lower():
                    # Extract model name from the full path (publishers/google/models/gemini-xxx)
                    models.append(model.name.split('/')[-1])
        except Exception as e:
            print(f"Warning: Could not fetch models from Vertex AI: {e}")
            return _cache_and_return(FALLBACK_MODELS)

        # Return fetched models or fallback
        return _cache_and_return(models if models else FALLBACK_MODELS)

    except Exception as e:
        print(f"Warning: Could not fetch models dynamically: {e}")
        return _cache_and_return(FALLBACK_MODELS)


@llm.hookimpl
def register_models(register):
    """
    Register available Vertex AI models with LLM.
    Attempts to fetch models dynamically from the API, falls back to hardcoded list.
    """
    models = get_available_models()

    for model in models:
        register(Vertex(f'vertex-{model}'))

    # TODO: How to register custom models?

class Vertex(llm.Model):
    model_id = ""
    model_name = ""
    can_stream = True

    class Options(llm.Options):
        max_output_tokens: Optional[int] = None
        temperature: Optional[float] = None
        top_p: Optional[float] = None
        top_k: Optional[int] = None

    def __init__(self, model_id):
        self.model_id = model_id
        self.model_name = model_id.replace('vertex-', '')

        # TODO: Can we save these with llm keys set or something instead?
        project_id = os.getenv('VERTEX_PROJECT_ID')
        location = os.getenv('VERTEX_LOCATION')
        vertexai.init(project=project_id, location=location)

    def execute(self, prompt, stream, response, conversation):
        self.model = GenerativeModel(model_name=self.model_name,
                                     system_instruction=[prompt.system] if prompt.system else None)
        history = self.build_history(conversation)
        chat = self.model.start_chat(history=history)
        responses = chat.send_message(prompt.prompt,
                                      stream=stream,
                                      generation_config=self.build_generation_config(prompt.options))
        if stream:
            for chunk in responses:
                yield chunk.text
        else:
            msg = responses.text
            yield msg

    def build_history(self, conversation):
        if not conversation:
            return []
        messages = []
        print(f"Build_history conversation: {conversation}")
        for response in conversation.responses:
            user_content = Content(role="user", parts=[Part.from_text(response.prompt.prompt)])
            model_content = Content(role="model", parts=[Part.from_text(response.text())])
            messages.extend([user_content, model_content])
        return messages

    def build_generation_config(self, options):
        return GenerationConfig(**options.model_dump())
