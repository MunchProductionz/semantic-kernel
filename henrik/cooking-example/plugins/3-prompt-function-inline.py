import semantic_kernel as sk
import os
import asyncio
from dotenv import load_dotenv
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAIChatPromptExecutionSettings
from semantic_kernel.functions import KernelArguments
from semantic_kernel.prompt_template import InputVariable, PromptTemplateConfig
load_dotenv()

# Definitions
prompt = """{{$input}}
Summarize the content above.
"""
ai_model_id = os.getenv("OPENAI_CHAT_MODEL_ID")

# Setup
kernel = sk.Kernel()
service_id = "default"
kernel.add_service(
    OpenAIChatCompletion(
        service_id=service_id,
        ai_model_id=ai_model_id,
    ),
)

# Prompt Function
execution_settings = OpenAIChatPromptExecutionSettings(
        service_id=service_id,
        ai_model_id=ai_model_id,
        max_tokens=2000,
        temperature=0.7,
    )
prompt_template_config = PromptTemplateConfig(
    template=prompt,
    name="summarize",
    template_format="semantic-kernel",
    input_variables=[
        InputVariable(name="input", description="The user input", is_required=True),
    ],
    execution_settings=execution_settings,
)

# Add the function - This is not using a specified file such as "Cooking -> RecipeGenerator"
summarize = kernel.add_function(
    prompt=prompt,
    function_name="summarizeFunc",
    plugin_name="summarizePlugin",
    description="Summarize the content",
    prompt_template_config=prompt_template_config,
)


# Testing
input_text = """
Demo (ancient Greek poet)
From Wikipedia, the free encyclopedia
Demo or Damo (Greek: Δεμώ, Δαμώ; fl. c. AD 200) was a Greek woman of the Roman period, known for a single epigram, engraved upon the Colossus of Memnon, which bears her name. She speaks of herself therein as a lyric poetess dedicated to the Muses, but nothing is known of her life.[1]
Identity
Demo was evidently Greek, as her name, a traditional epithet of Demeter, signifies. The name was relatively common in the Hellenistic world, in Egypt and elsewhere, and she cannot be further identified. The date of her visit to the Colossus of Memnon cannot be established with certainty, but internal evidence on the left leg suggests her poem was inscribed there at some point in or after AD 196.[2]
Epigram
There are a number of graffiti inscriptions on the Colossus of Memnon. Following three epigrams by Julia Balbilla, a fourth epigram, in elegiac couplets, entitled and presumably authored by "Demo" or "Damo" (the Greek inscription is difficult to read), is a dedication to the Muses.[2] The poem is traditionally published with the works of Balbilla, though the internal evidence suggests a different author.[1]
In the poem, Demo explains that Memnon has shown her special respect. In return, Demo offers the gift for poetry, as a gift to the hero. At the end of this epigram, she addresses Memnon, highlighting his divine status by recalling his strength and holiness.[2]
Demo, like Julia Balbilla, writes in the artificial and poetic Aeolic dialect. The language indicates she was knowledgeable in Homeric poetry—'bearing a pleasant gift', for example, alludes to the use of that phrase throughout the Iliad and Odyssey.[a][2]
"""

async def get_summary(input):
    summary = await kernel.invoke(summarize, input=input)
    print(summary)
    
def main():
    asyncio.run(get_summary(input_text))
    
main()