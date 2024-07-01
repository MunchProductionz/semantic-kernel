import semantic_kernel as sk
import os
import sys
import asyncio
from dotenv import load_dotenv
from semantic_kernel.connectors.ai.hugging_face import HuggingFaceTextCompletion, HuggingFaceTextEmbedding
from semantic_kernel.core_plugins import TextMemoryPlugin
from semantic_kernel.memory import SemanticTextMemory, VolatileMemoryStore
from semantic_kernel.connectors.ai.hugging_face import HuggingFacePromptExecutionSettings
from semantic_kernel.prompt_template import PromptTemplateConfig
load_dotenv()

# Get paths
notebook_dir = os.path.abspath("")
parent_dir = os.path.dirname(notebook_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)

# Setup
kernel = sk.Kernel()
collection_id = "generic"
text_service_id = "HuggingFaceM4/tiny-random-LlamaForCausalLM"          # Feel free to update this model to any other model available on Hugging Face
embed_service_id = "sentence-transformers/all-MiniLM-L6-v2"             # Feel free to update this model to any other model available on Hugging Face

# Add services (chat and text embedding)
kernel.add_service(
    service=HuggingFaceTextCompletion(
        service_id=text_service_id, ai_model_id=text_service_id, task="text-generation"
    ),
)
embedding_svc = HuggingFaceTextEmbedding(service_id=embed_service_id, ai_model_id=embed_service_id)
kernel.add_service(
    service=embedding_svc,
)
memory = SemanticTextMemory(storage=VolatileMemoryStore(), embeddings_generator=embedding_svc)
kernel.add_plugin(TextMemoryPlugin(memory), "TextMemoryPlugin")

# Add memories
async def populate_memory(memory: SemanticTextMemory) -> None:
    await memory.save_information(collection=collection_id, id="info1", text="Sharks are fish.")
    await memory.save_information(collection=collection_id, id="info2", text="Whales are mammals.")
    await memory.save_information(collection=collection_id, id="info3", text="Penguins are birds.")
    await memory.save_information(collection=collection_id, id="info4", text="Dolphins are mammals.")
    await memory.save_information(collection=collection_id, id="info5", text="Flies are insects.")

# Define and add prompt function
my_prompt = """I know these animal facts: 
- {{recall 'fact about sharks'}}
- {{recall 'fact about whales'}} 
- {{recall 'fact about penguins'}} 
- {{recall 'fact about dolphins'}} 
- {{recall 'fact about flies'}}
Now, tell me something about: {{$request}}"""

execution_settings = HuggingFacePromptExecutionSettings(
    service_id=text_service_id,
    ai_model_id=text_service_id,
    max_tokens=45,
    temperature=0.5,
    top_p=0.5,
)

prompt_template_config = PromptTemplateConfig(
    template=my_prompt,
    name="text_complete",
    template_format="semantic-kernel",
    execution_settings=execution_settings,
)

my_function = kernel.add_function(
    function_name="text_complete",
    plugin_name="TextCompletionPlugin",
    prompt_template_config=prompt_template_config,
)

## Testing
async def main():
    
    await populate_memory(memory)
    output = await kernel.invoke(
        my_function,
        request="What are whales?",
    )

    output = str(output).strip()

    query_result1 = await memory.search(
        collection=collection_id, query="What are sharks?", limit=1, min_relevance_score=0.3
    )

    print(f"The queried result for 'What are sharks?' is {query_result1[0].text}")

    print(f"{text_service_id} completed prompt with: '{output}'")
    
asyncio.run(main())