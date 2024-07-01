import semantic_kernel as sk
import os
import sys
import asyncio
from dotenv import load_dotenv
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import OpenAIChatCompletion
from semantic_kernel.connectors.ai.open_ai.services.open_ai_text_embedding import OpenAITextEmbedding
from semantic_kernel.core_plugins.text_memory_plugin import TextMemoryPlugin
from semantic_kernel.kernel import Kernel
from semantic_kernel.memory.semantic_text_memory import SemanticTextMemory
from semantic_kernel.memory.volatile_memory_store import VolatileMemoryStore
from semantic_kernel.functions import KernelFunction
from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.functions import KernelArguments
load_dotenv()

# Get paths
notebook_dir = os.path.abspath("")
parent_dir = os.path.dirname(notebook_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)

# Definitions
ask = """
Tomorrow is Valentine's day. I need to come up with a few short poems.
She likes Shakespeare so write using his style. She speaks French so write it in French.
Convert the text to uppercase."""

# Setup
kernel = Kernel()
chat_service_id = "chat"
collection_id = "generic"

# Add services (chat and text embedding)
oai_chat_service = OpenAIChatCompletion(
        service_id=chat_service_id,
)
embedding_gen = OpenAITextEmbedding(
    ai_model_id="text-embedding-3-small",
)
kernel.add_service(oai_chat_service)
kernel.add_service(embedding_gen)

# Define memory
memory = SemanticTextMemory(storage=VolatileMemoryStore(), embeddings_generator=embedding_gen)

# Add plugin
kernel.add_plugin(TextMemoryPlugin(memory), "TextMemoryPlugin")


# Manually adding memories
async def populate_memory(memory: SemanticTextMemory) -> None:
    # Add some documents to the semantic memory
    await memory.save_information(collection=collection_id, id="info1", text="Your budget for 2024 is $100,000")
    await memory.save_information(collection=collection_id, id="info2", text="Your savings from 2023 are $50,000")
    await memory.save_information(collection=collection_id, id="info3", text="Your investments are $80,000")

# asyncio.run(populate_memory(memory))

# Search the memory
async def search_memory_examples(memory: SemanticTextMemory) -> None:
    questions = [
        "What is my budget for 2024?",
        "What are my savings from 2023?",
        "What are my investments?",
    ]

    for question in questions:
        print(f"Question: {question}")
        result = await memory.search(collection_id, question)
        print(f"Answer: {result[0].text}\n")
        
# asyncio.run(search_memory_examples(memory))


## Using Memory in chat    -    'recall' takes an input ask and performs a similarity search on the contents that have been embedded in the Memory Store and returns the most relevant memory.
async def setup_chat_with_memory(
    kernel: Kernel,
    service_id: str,
) -> KernelFunction:
    prompt = """
    ChatBot can have a conversation with you about any topic.
    It can give explicit instructions or say 'I don't know' if
    it does not have an answer.

    Information about me, from previous conversations:
    - {{recall 'budget by year'}} What is my budget for 2024?
    - {{recall 'savings from previous year'}} What are my savings from 2023?
    - {{recall 'investments'}} What are my investments?

    {{$request}}
    """.strip()

    prompt_template_config = PromptTemplateConfig(
        template=prompt,
        execution_settings={
            service_id: kernel.get_service(service_id).get_prompt_execution_settings_class()(service_id=service_id)
        },
    )

    chat_func = kernel.add_function(
        function_name="chat_with_memory",
        plugin_name="chat",
        prompt_template_config=prompt_template_config,
    )

    return chat_func

# RelevanceParam is used in memory search and is a measure of the relevance score from 0.0 to 1.0, where 1.0 means a perfect match. We encourage users to experiment with different values.


## Testing
def main():
    
    # Setup
    kernel = Kernel()
    chat_service_id = "chat"

    # Add services, memory and plugins
    oai_chat_service = OpenAIChatCompletion(
            service_id=chat_service_id,
    )
    embedding_gen = OpenAITextEmbedding(
        ai_model_id="text-embedding-3-small",
    )
    kernel.add_service(oai_chat_service)
    kernel.add_service(embedding_gen)

    memory = SemanticTextMemory(storage=VolatileMemoryStore(), embeddings_generator=embedding_gen)

    kernel.add_plugin(TextMemoryPlugin(memory), "TextMemoryPlugin")
    
    
    print("Populating memory...")
    asyncio.run(populate_memory(memory))

    print("Asking questions... (manually)")
    asyncio.run(search_memory_examples(memory))

    print("Setting up a chat (with memory!)")
    chat_func = asyncio.run(setup_chat_with_memory(kernel, chat_service_id))

    print("Begin chatting (type 'exit' to exit):\n")
    print(
        "Welcome to the chat bot!\
        \n  Type 'exit' to exit.\
        \n  Try asking a question about your finances (i.e. \"talk to me about my finances\")."
    )

    async def chat(user_input: str):
        print(f"User: {user_input}")
        answer = await kernel.invoke(chat_func, request=user_input)
        print(f"ChatBot:> {answer}")
        
    asyncio.run(chat("What is my budget for 2024?"))
    asyncio.run(chat("talk to me about my finances"))
        
# main()
# asyncio.run(populate_memory(memory))


## Add external documents to memory

# GitHub - Semantic Kernel
github_files = {}
github_files["https://github.com/microsoft/semantic-kernel/blob/main/README.md"] = (
    "README: Installation, getting started, and how to contribute"
)
github_files[
    "https://github.com/microsoft/semantic-kernel/blob/main/dotnet/notebooks/02-running-prompts-from-file.ipynb"
] = "Jupyter notebook describing how to pass prompts from a file to a semantic plugin or function"
github_files["https://github.com/microsoft/semantic-kernel/blob/main/dotnet/notebooks/00-getting-started.ipynb"] = (
    "Jupyter notebook describing how to get started with the Semantic Kernel"
)
github_files["https://github.com/microsoft/semantic-kernel/tree/main/samples/plugins/ChatPlugin/ChatGPT"] = (
    "Sample demonstrating how to create a chat plugin interfacing with ChatGPT"
)
github_files[
    "https://github.com/microsoft/semantic-kernel/blob/main/dotnet/src/SemanticKernel/Memory/Volatile/VolatileMemoryStore.cs"
] = "C# class that defines a volatile embedding store"

# Add files to VolatileMemoryStore
memory_collection_name = "SKGitHub"
print("Adding some GitHub file URLs and their descriptions to a volatile Semantic Memory.")

for index, (entry, value) in enumerate(github_files.items()):
    asyncio.run(memory.save_reference(
        collection=memory_collection_name,      # Separate these memories from the chat memories by putting them in a different collection.
        description=value,
        text=value,
        external_id=entry,
        external_source_name="GitHub",
    ))
    print("  URL {} saved".format(index))
    
ask = "I love Jupyter notebooks, how should I get started?"
print("===========================\n" + "Query: " + ask + "\n")

memories = asyncio.run(memory.search(memory_collection_name, ask, limit=5, min_relevance_score=0.6))

for index, memory in enumerate(memories):
    print(f"Result {index}:")
    print("  URL:     : " + memory.id)
    print("  Title    : " + memory.description)
    print("  Relevance: " + str(memory.relevance))
    print()
    


## Add ecternal storage using Azure AI Search (if memory uses too much RAM)
# from semantic_kernel.connectors.memory.azure_cognitive_search import AzureCognitiveSearchMemoryStore
# api_key = os.getenv("AZURE_AI_SEARCH_API_KEY")
# endpoint = os.getenv("AZURE_AI_SEARCH_ENDPOINT")

# acs_memory_store = AzureCognitiveSearchMemoryStore(vector_size=1536, admin_key=api_key, search_endpoint=endpoint)   # Instead of VolatileMemoryStore
# memory = SemanticTextMemory(storage=acs_memory_store, embeddings_generator=embedding_gen)
# kernel.add_plugin(TextMemoryPlugin(memory), "TextMemoryPluginACS")
# asyncio.run(populate_memory(memory))
# asyncio.run(search_memory_examples(memory))