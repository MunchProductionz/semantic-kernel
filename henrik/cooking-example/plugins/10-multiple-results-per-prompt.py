import semantic_kernel as sk
import os
import time
import sys
import asyncio
from dotenv import load_dotenv
from semantic_kernel.connectors.ai.open_ai import (
    OpenAIChatCompletion,
    OpenAIChatPromptExecutionSettings,
    OpenAITextCompletion,
    OpenAITextPromptExecutionSettings
)
from semantic_kernel.connectors.ai.hugging_face import HuggingFacePromptExecutionSettings, HuggingFaceTextCompletion
from semantic_kernel.contents import ChatHistory
from IPython.display import clear_output
load_dotenv()

# Get paths
notebook_dir = os.path.abspath("")
parent_dir = os.path.dirname(notebook_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)

# Setup
kernel = sk.Kernel()
plugins_directory = "../plugins"

# Add OpenAI chat service
oai_chat_service_id = "oai_chat"
oai_chat_service = OpenAIChatCompletion(
    service_id=oai_chat_service_id,
)
oai_chat_prompt_execution_settings = OpenAIChatPromptExecutionSettings(
    service_id=oai_chat_service_id,
    max_tokens=80,
    temperature=0.7,
    top_p=1,
    frequency_penalty=0.5,
    presence_penalty=0.5,
    number_of_responses=3,
)

# Add OpenAI text service
oai_text_service_id = "oai_text"
oai_text_service = OpenAITextCompletion(
    service_id=oai_text_service_id,
)
oai_text_prompt_execution_settings = OpenAITextPromptExecutionSettings(
    service=oai_text_service_id,
    extension_data={
        "max_tokens": 80,
        "temperature": 0.7,
        "top_p": 1,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.5,
        "number_of_responses": 3,
    },
)

# Add Hugging Face text service
hf_text_service_id = "hf_text"
hf_text_service = HuggingFaceTextCompletion(
    service_id=hf_text_service_id,
    ai_model_id="distilgpt2",
    task="text-generation"
)
hf_prompt_execution_settings = HuggingFacePromptExecutionSettings(
    service_id=hf_text_service_id,
    extension_data={
        "max_new_tokens": 80,
        "temperature": 0.7,
        "top_p": 1,
        "num_return_sequences": 3
    },
)

## Multiple Results per Prompt
### Multiple Open AI Text Completions
async def get_openai_text_completions(prompt):
    results = oai_text_service.get_text_contents(
        prompt=prompt,
        settings=oai_text_prompt_execution_settings
    )
    
    for i, result in enumerate(results):
        print(f"Result {i+1}: {result}")

### Multiple Hugging Face Text Completions
async def get_hf_text_completions(prompt):
    results = hf_text_service.get_text_contents(
        prompt=prompt,
        settings=hf_prompt_execution_settings
    )
    
    for i, result in enumerate(results):
        print(f"Result {i+1}: {result}")

### Multiple Open AI Chat Completions
async def get_openai_chat_completions(prompt):
    chat = ChatHistory()
    chat.add_user_message(prompt)
    results = oai_chat_service.get_chat_message_contents(
        chat_history=chat,
        settings=oai_chat_prompt_execution_settings
    )
    
    for i, result in enumerate(results):
        print(f"Result {i+1}: {result}")

## Streaming Multiple Results
async def stream_openai_chat_completions(prompt):
    # Determine the clear command based on OS
    clear_command = "cls" if os.name == "nt" else "clear"

    chat = ChatHistory()
    chat.add_user_message(prompt)

    stream = oai_chat_service.get_streaming_chat_message_contents(
        chat_history=chat, settings=oai_chat_prompt_execution_settings
    )
    number_of_responses = oai_chat_prompt_execution_settings.number_of_responses
    texts = [""] * number_of_responses

    last_clear_time = time.time()
    clear_interval = 0.5  # seconds

    # Note: there are some quirks with displaying the output, which sometimes flashes and disappears.
    # This could be influenced by a few factors specific to Jupyter notebooks and asynchronous processing.
    # The following code attempts to buffer the results to avoid the output flashing on/off the screen.

    async for results in stream:
        current_time = time.time()

        # Update texts with new results
        for result in results:
            texts[result.choice_index] += str(result)

        # Clear and display output at intervals
        if current_time - last_clear_time > clear_interval:
            clear_output(wait=True)
            for idx, text in enumerate(texts):
                print(f"Result {idx + 1}: {text}")
            last_clear_time = current_time

    print("----------------------------------------")


# TODO: Test this
def main():
    prompt_oai_text = "What is the purpose of a rubber duck?"
    prompt_hf_text = "The purpose of a rubber duck is"
    prompt_oai_chat = "It's a beautiful day outside, birds are singing, flowers are blooming. On days like these, kids like you..."
    
    asyncio.run(get_openai_text_completions(prompt_oai_text))
    print()
    asyncio.run(get_hf_text_completions(prompt_hf_text))
    print()
    asyncio.run(get_openai_chat_completions(prompt_oai_chat))
    print()
    
    # Uncomment to stream results
    asyncio.run(stream_openai_chat_completions(prompt_oai_chat))