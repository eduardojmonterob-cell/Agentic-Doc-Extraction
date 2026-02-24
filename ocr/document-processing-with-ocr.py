import os
import base64

from dotenv import load_dotenv
from PIL import Image


import pytesseract
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent,AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

## Create a OCR tool
@tool
def orc_read_document(image_path: str)-> str:
    """Reads and iamge from the given path and returns extracted text from usinf OCR"""
    try:
        text = pytesseract.image_to_string(Image.open(image_path))
        return text

    except Exception as e:
        return f"Error processing the image: {str(e)}"

ocr_text = orc_read_document.run("ocr/images/invoice.png")
print("Raw OCR Output:\n--------------------------\n", ocr_text)

# Use RegEx to extract information ==> verty difficult and tent to fail

# Create an Agent
load_dotenv(override=True)

# 1. Defining the list of tools
tools = [orc_read_document]

# 2. Set up the OpenAI GPT model
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=1
    )
# 3. OpenAI-compatible prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant for extracting information from documents."
            "You have access to the following tools:"
            "OCR tool to extract raw text from images"
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
# 4. Create a proper tool-calling agent
agent = create_openai_functions_agent(llm, tools, prompt)

#5. Set up the AgentExecutor to run the tool-enabled loop
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# The agent needs the path to the document and a clear instruction
task = """
Please process the document at 'ocr/images/invoice.png' using the OCR tool and extract the following information
in JSON format:
- tax
- total
"""
response = agent_executor.invoke({"input": task})