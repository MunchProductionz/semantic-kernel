import semantic_kernel as sk
import os
import sys
import asyncio
from dotenv import load_dotenv
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions import KernelArguments
load_dotenv()

# Get paths
notebook_dir = os.path.abspath("")
parent_dir = os.path.dirname(notebook_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)

# Definitions
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

# Get prompt from file
plugins_directory = "../plugins"
cooking_functions = kernel.add_plugin(
    parent_directory=plugins_directory,
    plugin_name="Cooking"
    )
cooking_function = cooking_functions["RecipeGenerator"]

# Testing
async def get_recipe(input):
    recipe = await kernel.invoke(
        function=cooking_function,
        input=input,
        format="recipe"
    )
    return recipe

def main():
    input = "Chicken Adobo Filipino Style"
    recipe = asyncio.run(get_recipe(input))
    print(recipe)
    
main()