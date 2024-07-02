import semantic_kernel as sk
import os
import sys
import asyncio
import random
from dotenv import load_dotenv
from typing import Annotated
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAIChatPromptExecutionSettings
from semantic_kernel.prompt_template import InputVariable, PromptTemplateConfig
from semantic_kernel.functions import kernel_function
load_dotenv()

# Get paths
notebook_dir = os.path.abspath("")
parent_dir = os.path.dirname(notebook_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)

## Kernel Functions with Annotated Parameters
# Setup
kernel = sk.Kernel()

# Add services
service_id = "default"
kernel.add_service(
    OpenAIChatCompletion(
        service_id=service_id,
    ),
)

# Define plugin class - Add min and max number
class GenerateNumberPlugin:
    """
    Description: Generate a number between a min and a max.
    """

    @kernel_function(
        name="GenerateNumber",
        description="Generate a random number between min and max",
    )
    def generate_number(
        self,
        min: Annotated[int, "the minimum number of paragraphs"],
        max: Annotated[int, "the maximum number of paragraphs"] = 10,
    ) -> Annotated[int, "the output is a number"]:
        """
        Generate a number between min-max
        Example:
            min="4" max="10" => rand(4,8)
        Args:
            min -- The lower limit for the random number generation
            max -- The upper limit for the random number generation
        Returns:
            int value
        """
        try:
            return str(random.randint(min, max))
        except ValueError as e:
            print(f"Invalid input {min} and {max}")
            raise e

# Add plugin
generate_number_plugin = kernel.add_plugin(GenerateNumberPlugin(), "GenerateNumberPlugin")

# Define prompt (add a language parameter to allow the our CorgiStory function to be written in a specified language)
prompt = """
Write a short story about two Corgis on an adventure.
The story must be:
- G rated
- Have a positive message
- No sexism, racism or other bias/bigotry
- Be exactly {{$paragraph_count}} paragraphs long
- Be written in this language: {{$language}}
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
    name="summarize",
    template_format="semantic-kernel",
    input_variables=[
        InputVariable(name="paragraph_count", description="The number of paragraphs", is_required=True),
        InputVariable(name="language", description="The language of the story", is_required=True),
    ],
    execution_settings=execution_settings,
)

corgi_story = kernel.add_function(
    function_name="CorgiStory",
    plugin_name="CorgiPlugin",
    prompt_template_config=prompt_template_config,
)

# Run the number generator
generate_number = generate_number_plugin["GenerateNumber"]
number_result = asyncio.run(generate_number(kernel, min=1, max=5))
num_paragraphs = number_result.value
print(f"Generating a corgi story {num_paragraphs} paragraphs long.")

# Run the story generator - Pass the output to the semantic story function
desired_language = "Spanish"
story = asyncio.run(corgi_story.invoke(kernel, paragraph_count=num_paragraphs, language=desired_language))
print(f"Generating a corgi story {num_paragraphs} paragraphs long in {desired_language}.")
print("=====================================================")
print(story)
print("\n\n\n")


## Calling Native Functions within a Semantic Function
# - We will make our CorgiStory semantic function call a native function GenerateNames which will return names for our Corgi characters.
# - We do this using the syntax {{plugin_name.function_name}}.

# Define plugin class - Generate names
class GenerateNamesPlugin:
    """
    Description: Generate character names.
    """

    # The default function name will be the name of the function itself, however you can override this
    # by setting the name=<name override> in the @kernel_function decorator. In this case, we're using
    # the same name as the function name for simplicity.
    @kernel_function(description="Generate character names", name="generate_names")
    def generate_names(self) -> str:
        """
        Generate two names.
        Returns:
            str
        """
        names = {"Hoagie", "Hamilton", "Bacon", "Pizza", "Boots", "Shorts", "Tuna"}
        first_name = random.choice(list(names))
        names.remove(first_name)
        second_name = random.choice(list(names))
        return f"{first_name}, {second_name}"

# Add plugin
generate_names_plugin = kernel.add_plugin(GenerateNamesPlugin(), plugin_name="GenerateNames")
generate_names = generate_names_plugin["generate_names"]    # Not used

# Define prompt (use GenerateNames.generate_names to generate names)
prompt = """
Write a short story about two Corgis on an adventure.
The story must be:
- G rated
- Have a positive message
- No sexism, racism or other bias/bigotry
- Be exactly {{$paragraph_count}} paragraphs long
- Be written in this language: {{$language}}
- The two names of the corgis are {{GenerateNames.generate_names}}
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
    name="corgi-new",
    template_format="semantic-kernel",
    input_variables=[
        InputVariable(name="paragraph_count", description="The number of paragraphs", is_required=True),
        InputVariable(name="language", description="The language of the story", is_required=True),
    ],
    execution_settings=execution_settings,
)

corgi_story = kernel.add_function(
    function_name="CorgiStoryUpdated",
    plugin_name="CorgiPluginUpdated",
    prompt_template_config=prompt_template_config,
)

# Run the number generator
number_result = asyncio.run(generate_number(kernel, min=1, max=5))
num_paragraphs = number_result.value

# Run the story generator - Pass the output to the semantic story function
desired_language = "French"
story = asyncio.run(corgi_story.invoke(kernel, paragraph_count=num_paragraphs, language=desired_language))
print(f"Generating a corgi story {num_paragraphs} paragraphs long in {desired_language}.")
print("=====================================================")
print(story)


## Recap
# A quick review of what we've learned here:
# - (1) We've learned how to create native and prompt functions and register them to the kernel
# - (2) We've seen how we can use Kernel Arguments to pass in more custom variables into our prompt
# - (2) We've seen how we can call native functions within a prompt.