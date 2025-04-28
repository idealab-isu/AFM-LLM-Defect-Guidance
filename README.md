# AI-AFM-Agent for Defect Detection in AFM Images and LLM-based AFM Assistant

## 📋 Overview

<img src="images/overview.png" alt="Overview" width="500"/>

This project implements a Unified User Interface for AFM Defect Classification and LLM-based AFM Assistant.


## 📝 Key Features

- **Defect Classification**: Utilizes a fine-tuned VGG16 model to classify AFM images into one of four categories:
  - `good_images`
  - `Imaging Artifact`
  - `Not Tracking`
  - `Tip Contamination`

- **LLM-based AFM Assistant**: Provides a multi-turn conversation interface for users to ask questions about AFM images and defects.
  - Supports OpenAI and Anthropic LLMs
  - Allows users to ask follow-up questions and get recommendations for action

## 🚀 Getting Started

### 📦 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv afm_llm
source afm_llm/bin/activate  # On Unix/macOS
# or
.\afm_llm\Scripts\activate  # On Windows
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your API keys:
```bash
OPENAI_API_KEY=<your_openai_api_key>
ANTHROPIC_API_KEY=<your_anthropic_api_key>
```

### 🏃‍♂️ Run the Streamlit app:
```bash
streamlit run app.py
```

# TODO

- [x] Generate 50 detailed questions related to AFM image defects focusing on multiple sample types, scanning parameters, and defect types.
- [x] Generate answers using GPT-4o, Claude-3.5-sonnet.
- [x] Generate answers using Gemini 2.0 Flash, Claude 3.7 sonnet and GPT-o3-mini models.
- [x] Setup Label Studio for evaluation of the answers.
- [x] Get evaluation scores from AFM experts for the answers.
- [x] Work on UI for AFM Conversational Chatbot.
- [ ] Work to make the UI better.