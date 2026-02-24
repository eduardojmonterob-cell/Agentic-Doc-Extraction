from paddleocr import PaddleOCR
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import numpy as np
import cv2
import os

# Load environment variables
from dotenv import load_dotenv
_ = load_dotenv(override=True)

# PaddleOCR Basics 
# Initialize the English OCR model
ocr = PaddleOCR(use_angle_cls=True, lang='en')

#0. Create a PaddleOCR Tool

@tool
def paddle_ocr_read_document(image_path: str) -> List[Dict[str, Any]]:
    """
    Reads an image from the given path and returns extracted text 
    with bounding boxes.
    
    Returns a list of dictionaries, each containing:
    - 'text': the recognized text string
    - 'bbox': bounding box coordinates [x_min, y_min, x_max, y_max]
    - 'confidence': recognition confidence score (if available)
    """
    try:
        result = ocr.predict(image_path)
        page = result[0]
        
        texts = page['rec_texts'] 
        boxes = page['dt_polys']         
        scores = page.get('rec_scores', [None] * len(texts))  
        
        extracted_items = []
        for text, box, score in zip(texts, boxes, scores):
            x_coords = [point[0] for point in box]
            y_coords = [point[1] for point in box]
            bbox = [min(x_coords), min(y_coords), max(x_coords), 
                    max(y_coords)]
            
            item = {
                'text': text,
                'bbox': bbox,
            }
            if score is not None:
                item['confidence'] = score
                
            extracted_items.append(item)
        
        return extracted_items
    
    except Exception as e:
        return [{"error": f"Error reading image: {e}"}]
                
# 1. Defining the list of tools
tools = [paddle_ocr_read_document]

# 2. Set up the OpenAI GPT model
llm = ChatOpenAI(
    model="gpt-5-mini",
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