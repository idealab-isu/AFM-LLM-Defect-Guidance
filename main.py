import os
import json
import pandas as pd
from dotenv import load_dotenv
import argparse

load_dotenv()

# from langchain.chat_models import ChatOpenAI, ChatAnthropic, ChatGooglePalm
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.schema.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from langchain.output_parsers.structured import StructuredOutputParser

AVAILABLE_MODELS = ['gpt4o', 'claude3', 'gemini']
API_KEYS = {
    'gpt4o': os.getenv("OPENAI_API_KEY"),
    'claude3': os.getenv("ANTHROPIC_API_KEY"),
    'gemini': os.getenv("GOOGLE_API_KEY")
}

def get_model(model):
    if model not in AVAILABLE_MODELS:
        raise ValueError(f"Invalid model. Available models: {AVAILABLE_MODELS}")

    api_key = API_KEYS[model]
    if not api_key:
        raise ValueError(f"API key not found for model: {model}")

    if model == 'gpt4o':
        return ChatOpenAI(model_name="gpt-4o", openai_api_key=api_key)
    elif model == 'claude3':
        return ChatAnthropic(model="claude-3-5-haiku-latest", anthropic_api_key=api_key)
    elif model == 'gemini':
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)

class AFMResponse(BaseModel):
    answer: str = Field(description="A breif and technical answer to the AFM question.")
    recommendations: list[str] = Field(description="Practical steps or suggestions.")
    
def main(model_name):
    # List of AFM-related questions from the benchmark questions
    questions = []
    with open('./benchmark_questions/AFM_LLM_Combined_50_Questions.json', 'r') as f:
        questions_data = json.load(f)

    for question in questions_data:
        questions.append(question['question'])

    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Invalid model. Available models: {AVAILABLE_MODELS}")

    # Set your API keys
    api_key = API_KEYS[model_name]
    if not api_key:
        raise ValueError(f"API key not found for model: {model_name}")

    # Initialize models
    model = get_model(model_name)
    
    # Expert system prompt
    # system_prompt = SystemMessage(
    #     content="You are an expert in Atomic Force Microscopy (AFM). You provide precise and insightful answers about AFM image defects,\
    #             scanning parameters, biological samples, and tip contamination. Keep responses concise and technical.")

    system_prompt = SystemMessage(
        content="You are an expert in atomic force microscopy (AFM). You are given a type of defect (Tip Contamination or Not Tracking) in AFM image. You need to help the user understand the defect and how to avoid it. Give brief, technical and correct answer for the user questions.")
                #I have compiled the questions from AFM users and I want you to answer them.        

    # Define the output parser
    # parser = StructuredOutputParser(pydantic_schema=AFMResponse)
    
    # Placeholder for responses
    results = []

    # Loop through each question and get responses from each model
    for q in questions[:5]:
        # format_instructions = parser.get_format_instructions()
        # prompt = f"""
        # {format_instructions}
        # Question: {q}
        # """
        prompt = q
        user_msg = HumanMessage(content=prompt)
        messages = [system_prompt, user_msg]

        # Get responses
        print(f"\n🧪 Question: {q}\n{'='*80}")
        model_with_structured_output = model.with_structured_output(AFMResponse)
        model_resp = model_with_structured_output.invoke(messages)
        # model_resp = model.invoke(messages).content
        # model_resp = parser.parse(model_resp)
        print(f"\n[{model_name}]:\n{model_resp}")
        # Save to list
        results.append({
            "Question": q,
            # "Model": model_name,
            "Answer": model_resp.answer,
            "Recommendations": model_resp.recommendations
        })

    # Export to JSON instead of CSV
    with open(f"afm_model_comparison_{model_name}.json", 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AFM Model Comparison')
    parser.add_argument('--model', type=str, default='gpt4o', choices=['gpt4o', 'claude3', 'gemini'], help='Model to use')
    args = parser.parse_args()

    main(args.model)