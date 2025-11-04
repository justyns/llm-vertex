# llm-vertex

Plugin for LLM adding support for Google Cloud Vertex AI.

Please note that this plugin is for Vertex AI specifically, not Google AI Studio.

For Gemini support using AI Studio, please see [llm-gemini](https://github.com/simonw/llm-gemini) instead.

Supported models:

- gemini-2.5-flash
- gemini-2.5-pro
- gemini-2.0-flash-lite
- gemini-2.0-flash
- gemini-1.5-pro
- gemini-1.5-flash

## Installation

See [Installing Plugins](https://llm.datasette.io/en/stable/plugins/installing-plugins.html) for detailed instructions.

**Method 1: Use llm**

``` shell
llm install llm-vertex
```

**Method 2: Use pip**

``` shell
pip install llm-vertex
```

## Use

First, authenticate using `gcloud`:

``` shell
gcloud auth application-default login
```

Export two environment variables for the GCP Project and location you want to use:

``` shell
export VERTEX_PROJECT_ID=gcp-project-id VERTEX_LOCATION=us-east1
```

Run llm and specify one of the provided models:

``` shell
❯ llm -m vertex-gemini-1.5-pro-preview-0409 "What's one clever name for a pet pelican?"
"Gulliver" would be a clever name for a pet pelican, referencing both its large gullet and its potential for long journeys! 🦜
```

## Development

Create and activate a virtual environment:

``` shell
python -m venv .venv
source .venv/bin/activate
```

Install the package in development mode with test dependencies:

``` shell
pip install -e '.[dev]'
```

Run the tests:

``` shell
python -m pytest
```

The tests use mocking to avoid requiring actual Google Cloud credentials during development, but do not really test actual functionality outside of making sure the plugin is installed and can be used.
