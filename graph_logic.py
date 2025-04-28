# --- START OF FILE graph_logic.py ---

import os
from typing import TypedDict, Annotated, List
import operator
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from langchain.schema import BaseMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver # Optional for checkpointing

# Load environment variables (optional here, but good practice if testing independently)
# load_dotenv() # Can be commented out if only app_main.py loads it

# --- LangGraph State Definition ---

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        messages: The list of messages comprising the conversation.
                  operator.add indicates messages should be appended.
    """
    messages: Annotated[List[BaseMessage], operator.add]

# --- LLM Initialization ---

def initialize_llm(provider: str, model_name: str, temperature: float, api_key: str):
    """Initializes the appropriate LangChain Chat Model."""
    if provider == "Groq":
        if not api_key:
            raise ValueError("Groq API key is missing. Please set GROQ_API_KEY.")
        return ChatGroq(api_key=api_key, model_name=model_name, temperature=temperature)
    elif provider == "OpenAI":
        if not api_key:
            raise ValueError("OpenAI API key is missing. Please set OPENAI_API_KEY.")
        return ChatOpenAI(api_key=api_key, model_name=model_name, temperature=temperature)
    elif provider == "Anthropic":
        if not api_key:
            raise ValueError("Anthropic API key is missing. Please set ANTHROPIC_API_KEY.")
        return ChatAnthropic(api_key=api_key, model_name=model_name, temperature=temperature)
    elif provider == "Google":
        if not api_key:
            raise ValueError("Google API key is missing. Please set GOOGLE_API_KEY.")
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

# --- LangGraph Node and Graph Building ---

def create_chat_graph(llm):
    """
    Builds and compiles the LangGraph conversational graph.

    Args:
        llm: An initialized LangChain Chat Model instance.

    Returns:
        A compiled LangGraph application.
    """

    # Define the function that calls the LLM - it closes over the 'llm' variable
    def call_model(state: GraphState) -> dict:
        """Invokes the provided LLM with the current conversation state."""
        messages = state['messages']
        response = llm.invoke(messages)
        # Return the AIMessage list to be added to the state
        return {"messages": [response]}

    # Build the graph workflow
    workflow = StateGraph(GraphState)

    # Add the single node that runs the LLM
    workflow.add_node("llm_node", call_model)

    # Set the entry point and the only edge
    workflow.set_entry_point("llm_node")
    workflow.add_edge("llm_node", END) # Conversation ends after one LLM call per turn

    # Compile the graph
    # Optional: Add memory for checkpointing if needed
    # memory = MemorySaver()
    # graph = workflow.compile(checkpointer=memory)
    graph = workflow.compile()

    return graph

# --- END OF FILE graph_logic.py ---