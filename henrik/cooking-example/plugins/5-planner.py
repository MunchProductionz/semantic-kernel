import semantic_kernel as sk
import os
import sys
import asyncio
from dotenv import load_dotenv
from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings
from semantic_kernel.core_plugins.text_plugin import TextPlugin
from semantic_kernel.functions.kernel_function_from_prompt import KernelFunctionFromPrompt
from semantic_kernel.kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAIChatPromptExecutionSettings
from semantic_kernel.contents.chat_history import ChatHistory  # noqa: F401
from semantic_kernel.functions.kernel_arguments import KernelArguments  # noqa: F401
from semantic_kernel.prompt_template.input_variable import InputVariable  # noqa: F401
from semantic_kernel.planners import SequentialPlanner
from typing import Annotated
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.core_plugins.math_plugin import MathPlugin
from semantic_kernel.core_plugins.time_plugin import TimePlugin
from semantic_kernel.planners.function_calling_stepwise_planner import (
    FunctionCallingStepwisePlanner,
    FunctionCallingStepwisePlannerOptions,
)
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
plugins_directory = "../plugins"
ai_model_id = os.getenv("OPENAI_CHAT_MODEL_ID")

## Setup
sequential_kernel = sk.Kernel()
sequential_service_id = "default"
sequential_kernel.add_service(
    OpenAIChatCompletion(
        service_id=sequential_service_id,
        ai_model_id=ai_model_id,
    ),
)

# Add plugins
summarize_plugin = sequential_kernel.add_plugin(
    plugin_name="SummarizePlugin",
    parent_directory=plugins_directory
)
writer_plugin = sequential_kernel.add_plugin(
    plugin_name="WriterPlugin",
    parent_directory=plugins_directory
)
# Add new plugin (not in any folder)
text_plugin = sequential_kernel.add_plugin(
    plugin=TextPlugin(),
    plugin_name="TextPlugin"
)

# Execution settings
prompt_execution_settings=OpenAIChatPromptExecutionSettings(
    service_id=sequential_service_id,
    max_tokens=2000,
    temperature=0.8,
),

## NOTE: KernelFunctionFromPrompt does not work due to Pydantic error
# Add the function - Prompt
# shakespeare_func = KernelFunctionFromPrompt(
#     function_name="Shakespeare",
#     plugin_name="WriterPlugin",
#     prompt="""
#         {{$input}}

#         Rewrite the above in the style of Shakespeare.
#         """,
#     prompt_execution_settings=prompt_execution_settings,
#     description="Rewrite the input in the style of Shakespeare.",
# )
# sequential_kernel.add_function(
#     plugin_name="WriterPlugin",
#     function=shakespeare_func
# )

# Print plugins and function
def print_plugins_and_functions(kernel):
    for plugin_name, plugin in kernel.plugins.items():
        for function_name, function in plugin.functions.items():
            print(f"Plugin: {plugin_name}, Function: {function_name}")
        

## Sequential Planner
sequential_planner = SequentialPlanner(sequential_kernel, sequential_service_id)   

async def create_sequential_plan(ask, planner):
    sequential_plan = await planner.create_plan(goal=ask)
    return sequential_plan
        
async def execute_sequential_plan(sequential_plan, kernel):
    result = await sequential_plan.invoke(kernel)
    return result
        
def print_sequential_planner_steps(sequential_plan):
    print("The plan's steps are:")
    for step in sequential_plan._steps:
        print(
            f"- {step.description.replace('.', '') if step.description else 'No description'} using {step.metadata.fully_qualified_name} with parameters: {step.parameters}"
        )
        
def test_sequential_planner(ask, kernel, planner):
    sequential_plan = asyncio.run(create_sequential_plan(ask, planner))
    print_sequential_planner_steps(sequential_plan)
    result = asyncio.run(execute_sequential_plan(sequential_plan, kernel))
    print(result)
    
    # Uncomment the following line to view the planner's process for completing the request
    # print_sequential_planner_steps(sequential_plan)


## Function Calling Stepwise Planner        NOTE: Uses OpenAI function calling. Can only use either the AzureChatCompletion or the OpenAIChatCompletion service.
# Steup
function_calling_stepwise_planner_kernel = Kernel()
function_calling_stepwise_planner_service_id = "default"
function_calling_stepwise_planner_kernel.add_service(
    OpenAIChatCompletion(
        service_id=function_calling_stepwise_planner_service_id,
    ),
)

# Make EmailPlugin
class EmailPlugin:
    """
    Description: EmailPlugin provides a set of functions to send emails.

    Usage:
        kernel.import_plugin_from_object(EmailPlugin(), plugin_name="email")

    Examples:
        {{email.SendEmail}} => Sends an email with the provided subject and body.
    """

    @kernel_function(name="SendEmail", description="Given an e-mail and message body, send an e-email")
    def send_email(
        self,
        subject: Annotated[str, "the subject of the email"],
        body: Annotated[str, "the body of the email"],
    ) -> Annotated[str, "the output is a string"]:
        """Sends an email with the provided subject and body."""
        return f"Email sent with subject: {subject} and body: {body}"

    @kernel_function(name="GetEmailAddress", description="Given a name, find the email address")
    def get_email_address(
        self,
        input: Annotated[str, "the name of the person"],
    ):
        email = ""
        if input == "Jane":
            email = "janedoe4321@example.com"
        elif input == "Paul":
            email = "paulsmith5678@example.com"
        elif input == "Mary":
            email = "maryjones8765@example.com"
        else:
            input = "johndoe1234@example.com"
        return email

# Add EmailPlugin    
function_calling_stepwise_planner_kernel.add_plugin(
    plugin=EmailPlugin(),
    plugin_name="EmailPlugin"
)

# Add other plugins
function_calling_stepwise_planner_kernel.add_plugin(
    plugin=MathPlugin(),
    plugin_name="MathPlugin"
)
function_calling_stepwise_planner_kernel.add_plugin(
    plugin=TimePlugin(),
    plugin_name="TimePlugin"
)

# Define questions
questions = [
    "What is the current hour number, plus 5?",
    "What is 387 minus 22? Email the solution to John and Mary.",
    "Write a limerick, translate it to Spanish, and send it to Jane",
]

# Define FunctionCallingStepwisePlanner
function_calling_stepwise_planner_options = FunctionCallingStepwisePlannerOptions(
    max_iterations=10,
    max_tokens=4000,
)
function_calling_stepwise_planner = FunctionCallingStepwisePlanner(service_id=function_calling_stepwise_planner_service_id, options=function_calling_stepwise_planner_options)

async def execute_stepwise_planner(question, kernel, planner):
    print("Executing stepwise planner...")
    result = await planner.invoke(kernel, question)
    return result

def print_stepwise_planner_process(result):
    return print(f"Chat history: {result.chat_history}\n")

async def test_function_calling_stepwise_planner(questions, kernel, planner):
    for question in questions:
        print(f"Q: {question}")
        result = await execute_stepwise_planner(question, kernel, planner)
        print(f"A: {result.final_answer}\n\n")

        # Uncomment the following line to view the planner's process for completing the request
        # print_stepwise_planner_process(result)

        
def main():
    test_sequential_planner(ask, sequential_kernel, sequential_planner)
    print("\n")
    asyncio.run(test_function_calling_stepwise_planner(questions, function_calling_stepwise_planner_kernel, function_calling_stepwise_planner))
            
main()