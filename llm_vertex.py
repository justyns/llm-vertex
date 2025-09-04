import llm
import vertexai
import os
from typing import Optional
# from google.cloud.aiplatform_v1beta1.types import Content, Part
from vertexai.generative_models import GenerativeModel, Part, ChatSession, Content, GenerationConfig


@llm.hookimpl
def register_models(register):
    # Source: https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models
    models = [
        'gemini-2.5-flash',
        'gemini-2.5-pro',
        'gemini-2.0-flash-lite',
        'gemini-2.0-flash',
        'gemini-1.5-pro',
        'gemini-1.5-flash',
    ]
    
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
