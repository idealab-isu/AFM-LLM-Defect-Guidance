import pandas as pd
# from langchain.chat_models import ChatOpenAI, ChatAnthropic, ChatGooglePalm
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.schema.messages import SystemMessage, HumanMessage

# Set your API keys
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
ANTHROPIC_API_KEY = "YOUR_ANTHROPIC_API_KEY"
GOOGLE_API_KEY = "YOUR_GEMINI_API_KEY"

# Initialize models
#gpt4o = ChatOpenAI(model_name="gpt-4o", openai_api_key=OPENAI_API_KEY)
#claude3 = ChatAnthropic(model="claude-3-opus-20240229", anthropic_api_key=ANTHROPIC_API_KEY)
gemini = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)

# Expert system prompt
system_prompt = SystemMessage(
    content="You are an expert in Atomic Force Microscopy (AFM). You provide precise and insightful answers about AFM image defects,\
             scanning parameters, biological samples, and tip contamination. Keep responses concise and technical.")

# system_prompt = SystemMessage(
#     content="You are an expert in atomic force microscopy (AFM). You are given a type of defect in AFM image.\
#             You need to help the user understand the defect and how to avoid it. Give concise but correct answer.\
#             I have compiled the questions from AFM users and I want you to answer them. Please output in the following format: Question and Answer")


# List of AFM-related questions
questions = [
    "What causes tip contamination in AFM images, and how can I prevent it when scanning biological samples?",
    "How can I detect if the AFM is not tracking the sample correctly?",
    "What scan parameters can I adjust to reduce noise in AFM images of soft materials?",
    "How does tapping mode help in imaging live cells?",
    "What are the signs of a worn-out AFM tip during a scan?"
]

# Placeholder for responses
results = []

# Loop through each question and get responses from each model
for q in questions:
    user_msg = HumanMessage(content=q)
    messages = [system_prompt, user_msg]

    # Get responses
    print(f"\n🧪 Question: {q}\n{'='*80}")
    gpt4o_resp = gpt4o.invoke(messages).content
    claude_resp = claude3.invoke(messages).content
    gemini_resp = gemini.invoke(messages).content
    
    # Save to list
    results.append({
        "Question": q,
        "GPT-4o": gpt4o_resp,
        "Claude 3 Opus": claude_resp,
        "Gemini Flash": gemini_resp
    })

# Convert to DataFrame and export
df = pd.DataFrame(results)
df.to_csv("afm_model_comparison.csv", index=False)

# Display in terminal
for i, row in df.iterrows():
    print(f"\nQuestion {i+1}: {row['Question']}")
    print(f"\n[GPT-4o]:\n{row['GPT-4o']}")
    print(f"\n[Claude 3 Opus]:\n{row['Claude 3 Opus']}")
    print(f"\n[Gemini Flash]:\n{row['Gemini Flash']}")
    print("\n" + "="*80)
