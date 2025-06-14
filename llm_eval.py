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
from langchain_groq import ChatGroq

from langchain.schema.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from langchain.output_parsers.structured import StructuredOutputParser

from langchain_core.rate_limiters import InMemoryRateLimiter

AVAILABLE_MODELS = ['gpt4o', 'gpt-o3-mini', 'claude3-5-sonnet', 'claude3-7-sonnet', 'gemini', 'groq']
API_KEYS = {
    'gpt4o': os.getenv("OPENAI_API_KEY"),
    'gpt-o3-mini': os.getenv("OPENAI_API_KEY"),
    'claude3-5-sonnet': os.getenv("ANTHROPIC_API_KEY"),
    'claude3-7-sonnet': os.getenv("ANTHROPIC_API_KEY"),
    'gemini': os.getenv("GOOGLE_API_KEY"),
    'groq': os.getenv("GROQ_API_KEY")
}


def get_model(model):
    if model not in AVAILABLE_MODELS:
        raise ValueError(f"Invalid model. Available models: {AVAILABLE_MODELS}")

    api_key = API_KEYS[model]
    if not api_key:
        raise ValueError(f"API key not found for model: {model}")

    rate_limiter = InMemoryRateLimiter(
        requests_per_second=0.1,  # <-- Super slow! We can only make a request once every 10 seconds!!
        check_every_n_seconds=0.1,  # Wake up every 100 ms to check whether allowed to make a request,
        max_bucket_size=10,  # Controls the maximum burst size.
    )
    if model == 'gpt4o':
        return ChatOpenAI(model_name="gpt-4o", openai_api_key=api_key, rate_limiter=rate_limiter)
    elif model == 'gpt-o3-mini':
        return ChatOpenAI(model_name="o3-mini", openai_api_key=api_key, rate_limiter=rate_limiter)
    elif model == 'claude3-5-sonnet':
        return ChatAnthropic(model="claude-3-5-sonnet-latest", anthropic_api_key=api_key)
    elif model == 'claude3-7-sonnet':
        return ChatAnthropic(model="claude-3-7-sonnet-latest", anthropic_api_key=api_key)
    elif model == 'gemini':
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key, rate_limiter=rate_limiter)
        #return ChatGoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25", google_api_key=api_key)
    elif model == 'groq':
        # return ChatGroq(model="llama3-8b-8192", groq_api_key=api_key)
        return ChatGroq(model="deepseek-r1-distill-llama-70b", groq_api_key=api_key, rate_limiter=rate_limiter)

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
    for idx, q in enumerate(questions):
        # format_instructions = parser.get_format_instructions()
        # prompt = f"""
        # {format_instructions}
        # Question: {q}
        # """
        prompt = q
        user_msg = HumanMessage(content=prompt)
        messages = [system_prompt, user_msg]

        # Get responses
        print(f"\n🧪 Question {idx+1}: {q}\n{'='*80}")
        model_with_structured_output = model.with_structured_output(AFMResponse)
        model_resp = model_with_structured_output.invoke(messages)
        # model_resp = model.invoke(messages).content
        # model_resp = parser.parse(model_resp)
        print(f"\n[{model_name}]:\n{model_resp}")
        # Save to list
        result = {
            "idx": idx,
            "Question": q,
            # "Model": model_name,
            "Answer": model_resp.answer,
            "Recommendations": model_resp.recommendations
        }
        results.append(result)
        
    # Export to JSON instead of CSV
    with open(f"./llm_responses/responses_from_{model_name}.json", 'w') as f:
       json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AFM Model Comparison')
    parser.add_argument('--model', type=str, default='gpt4o', choices=['gpt4o', 'gpt-o3-mini', 'claude3-5-sonnet', 'claude3-7-sonnet', 'gemini', 'groq'], help='Model to use')
    args = parser.parse_args()
    import time
    start_time = time.time()
    main(args.model)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    # write the time taken to a file
    with open(f"./llm_responses/time_taken_{args.model}.txt", 'w') as f:
        f.write(f"Total time taken for 10 questions: {end_time - start_time} seconds")