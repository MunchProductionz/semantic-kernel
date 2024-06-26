import semantic_kernel as sk
import os
import asyncio
from dotenv import load_dotenv
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions import KernelArguments
load_dotenv()

# Setup
kernel = sk.Kernel()
api_key = os.getenv("OPENAI_API_KEY")
prompt = "Chicken Adobo Filipino Style"
plugins_directory = "../plugins"
ai_model_id = os.getenv("OPENAI_CHAT_MODEL_ID")

# OpenAI Service
kernel.add_service(OpenAIChatCompletion(
    ai_model_id=ai_model_id,
    service_id="default",
    api_key=api_key
    ))

# Cooking Plugin
cooking_skill = kernel.add_plugin(
    parent_directory=plugins_directory,
    plugin_name="Cooking"
    )
recipe_function = cooking_skill["RecipeGenerator"]

# Marketing Plugin
advertisement_skill = kernel.add_plugin(
    parent_directory=plugins_directory,
    plugin_name="Marketing"
    )
advertisment_function = advertisement_skill["AdvertisementGenerator"]


## With async
async def get_recipe(prompt):
    
    recipe = await kernel.invoke(
        recipe_function,
        KernelArguments(
            input=prompt,
            format=""
            )
    )
    return recipe

async def get_advertisement(recipe):
    advertisement = await kernel.invoke(
        advertisment_function,
        KernelArguments(
            input=recipe
            )
    )
    return advertisement


### With main function
async def main():
    
    # load_dotenv()
    recipe = await get_recipe(prompt)
    advertisement = await get_advertisement(recipe)

    print("Recipe: ", recipe)
    print()
    print("Advertisement: ", advertisement)

asyncio.run(main())

### Without main function
# recipe = asyncio.run(get_recipe(prompt))
# advertisement = asyncio.run(get_advertisement(recipe))

# print("Recipe: ", recipe)
# print()
# print("Advertisement: ", advertisement)



## Without async

# Results
# recipe_result = recipe_function(prompt)
# advertisement_result = advertisment_function(recipe_result)
# print("Recipe: ", recipe_result)
# print("Advertisement: ", advertisement_result)

# recipe = await kernel.invoke(
#     recipe_function,
#     KernelArguments(
#         input=prompt,
#         format=""
#         )    
# )
# recipe = await kernel.invoke(recipe_function, input=prompt, format="")     # Alternative
# print(recipe)