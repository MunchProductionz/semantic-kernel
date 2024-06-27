import semantic_kernel as sk
import os
import sys
import asyncio
from dotenv import load_dotenv
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAIChatPromptExecutionSettings
from semantic_kernel.functions import KernelArguments
from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.prompt_template.input_variable import InputVariable
from semantic_kernel.contents import ChatHistory
load_dotenv()

# Get paths
notebook_dir = os.path.abspath("")
parent_dir = os.path.dirname(notebook_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)

# Definitions
prompt = """
ChatBot can have a conversation with you about any topic.
It can give explicit instructions or say 'I don't know' if it does not have an answer.

{{$history}}
User: {{$user_input}}
ChatBot: """
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
    name="chat",
    template_format="semantic-kernel",
    input_variables=[
        InputVariable(name="user_input", description="The user input", is_required=True),
        InputVariable(name="history", description="The conversation history", is_required=True),
    ],
    execution_settings=execution_settings,
)

# Add the function - This is not using a specified file such as "Cooking -> RecipeGenerator"
chat_function = kernel.add_function(
    function_name="chat",
    plugin_name="chatPlugin",
    prompt_template_config=prompt_template_config,
)

# Initialize the chat history
chat_history = ChatHistory()
chat_history.add_system_message("You are a helpful chatbot who is good about giving book recommendations.")

# Chat with the Bot
async def add_first_message(user_input: str) -> None:
    arguments = KernelArguments(user_input="Hi, I'm looking for book suggestions", history=chat_history)
    response = await kernel.invoke(chat_function, arguments)
    chat_history.add_user_message(user_input)
    print(response)

async def chat(input_text: str) -> None:
    # Save new message in the context variables
    print(f"User: {input_text}")

    # Process the user message and get an answer
    answer = await kernel.invoke(chat_function, KernelArguments(user_input=input_text, history=chat_history))

    # Show the response
    print(f"ChatBot: {answer}")

    chat_history.add_user_message(input_text)
    chat_history.add_assistant_message(str(answer))
    
def main():
    asyncio.run(add_first_message("Hi, I'm looking for book suggestions"))
    asyncio.run(chat("I love history and philosophy, I'd like to learn something new about Greece, any suggestion?"))
    asyncio.run(chat("that sounds interesting, what is it about?"))
    asyncio.run(chat("if I read that book, what exactly will I learn about Greek history?"))
    asyncio.run(chat("could you list some more books I could read about this topic?"))
    
    print(chat_history)
    
    # while True:
    #     input_text = input("User: ")
    #     if input_text.lower() == "exit":
    #         break
    #     asyncio.run(chat(input_text))         # Alternative
    
main()