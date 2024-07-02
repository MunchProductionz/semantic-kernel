import semantic_kernel as sk
import os
import sys
import asyncio
import random
from dotenv import load_dotenv
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAIChatPromptExecutionSettings
from semantic_kernel.prompt_template import InputVariable, PromptTemplateConfig
from semantic_kernel.functions import kernel_function
load_dotenv()

# Get paths
notebook_dir = os.path.abspath("")
parent_dir = os.path.dirname(notebook_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)

# Setup
kernel = sk.Kernel()

# Add services
service_id = "default"
kernel.add_service(
    OpenAIChatCompletion(
        service_id=service_id,
    ),
)

# Define plugin class
class GenerateNumberPlugin:
    """
    Description: Generate a number between 3-x.
    """

    @kernel_function(
        name="GenerateNumberThreeOrHigher",
        description="Generate a random number between 3-x",
    )
    def generate_number_three_or_higher(
        self,
        input: str
    ) -> str:
        """
        Generate a number between 3-<input>
        Example:
            "8" => rand(3,8)
        Args:
            input -- The upper limit for the random number generation
        Returns:
            int value
        """
        try:
            return str(random.randint(3, int(input)))
        except ValueError as e:
            print(f"Invalid input {input}")
            raise e
        
# Define prompt
prompt = """
Write a short story about two Corgis on an adventure.
The story must be:
- G rated
- Have a positive message
- No sexism, racism or other bias/bigotry
- Be exactly {{$input}} paragraphs long. It must be this length.
"""

# Add function
execution_settings = OpenAIChatPromptExecutionSettings(
        service_id=service_id,
        ai_model_id="gpt-3.5-turbo",
        max_tokens=2000,
        temperature=0.7,
    )

prompt_template_config = PromptTemplateConfig(
    template=prompt,
    name="story",
    template_format="semantic-kernel",
    input_variables=[
        InputVariable(name="input", description="The user input", is_required=True),
    ],
    execution_settings=execution_settings,
)

corgi_story = kernel.add_function(
    function_name="CorgiStory",
    plugin_name="CorgiPlugin",
    prompt_template_config=prompt_template_config,
)

# Add plugin
generate_number_plugin = kernel.add_plugin(GenerateNumberPlugin(), "GenerateNumberPlugin")

# Run the number generator
generate_number_three_or_higher = generate_number_plugin["GenerateNumberThreeOrHigher"]
number_result = asyncio.run(generate_number_three_or_higher(kernel, input=6))
print(number_result)

# Run the story generator
story = asyncio.run(corgi_story.invoke(kernel, input=number_result.value))
print(f"Generating a corgi story exactly {number_result.value} paragraphs long.")
print("=====================================================")
print(story)