import llm
import os
from typing import Optional
from google import genai
from google.genai import types


@llm.hookimpl
def register_models(register):
    # Source: https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models
    models = [
        'gemini-1.5-flash',
        'gemini-1.5-pro',
        'gemini-2.0-flash',
        'gemini-2.0-flash-lite',
        'gemini-2.5-flash',
        'gemini-2.5-flash-lite',
        'gemini-2.5-pro',
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
        self.client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location
        )

    def execute(self, prompt, stream, response, conversation):
        config = types.GenerateContentConfig(
            system_instruction=prompt.system if prompt.system else None,
            temperature=prompt.options.temperature,
            max_output_tokens=prompt.options.max_output_tokens,
            top_p=prompt.options.top_p,
            top_k=prompt.options.top_k
        )

        history = self.build_history(conversation)
        chat = self.client.chats.create(
            model=self.model_name,
            config=config,
            history=history if history else None
        )

        if stream:
            for chunk in chat.send_message_stream(prompt.prompt):
                yield chunk.text
        else:
            response_msg = chat.send_message(prompt.prompt)
            yield response_msg.text

    def build_history(self, conversation):
        if not conversation:
            return []
        messages = []
        print(f"Build_history conversation: {conversation}")
        for response in conversation.responses:
            user_content = types.Content(
                role="user",
                parts=[types.Part(text=response.prompt.prompt)]
            )
            model_content = types.Content(
                role="model",
                parts=[types.Part(text=response.text())]
            )
            messages.extend([user_content, model_content])
        return messages
