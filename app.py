# --- START OF FILE app_main.py ---

import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
# Ensure classification_model is importable (e.g., it's in the same directory orPYTHONPATH)
try:
    from classification_model.model import VGG16Model
except ImportError:
    st.error("Could not import VGG16Model. Make sure 'classification_model/model.py' exists and is importable.")
    st.stop()

from PIL import Image
import os
from dotenv import load_dotenv
from langchain.schema import SystemMessage, HumanMessage, AIMessage, BaseMessage
import io

# Import graph logic from the other file
try:
    from graph_logic import initialize_llm, create_chat_graph, GraphState
except ImportError:
     st.error("Could not import from graph_logic.py. Make sure the file exists in the same directory.")
     st.stop()


# Load environment variables from .env file
load_dotenv()

# --- Helper Functions (Specific to App/UI) ---

@st.cache_resource
def load_vgg16(path='./classification_model/best_model.pth', device='cpu'):
    """Loads the pre-trained VGG16 model."""
    model = VGG16Model()
    if not os.path.exists(path):
        st.error(f"Model file not found at {path}. Please ensure the model is present.")
        st.stop()
    try:
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
    except Exception as e:
        st.error(f"Error loading model state dict: {e}")
        st.stop()
    return model

@st.cache_resource
def load_class_labels() -> dict:
    """Loads the class labels."""
    return {0: 'good_images', 1: 'Imaging Artifact', 2: 'Not Tracking', 3: 'Tip Contamination'}

def preprocess_image(img: Image.Image):
    """Preprocesses the image for the VGG16 model."""
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3718, 0.1738, 0.0571], std=[0.2095, 0.2124, 0.1321]),
    ])
    img_tensor = preprocess(img).unsqueeze(0)
    return img_tensor

def predict_image_class(img: Image.Image, model, class_names: dict) -> str:
    """Predicts the class label for a given image."""
    img_tensor = preprocess_image(img)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        top_prob, top_idx = torch.topk(probs, 1)
        class_label = class_names.get(top_idx.item(), "Unknown Class")
    return class_label

def download_chat_history() -> str:
    """Generates the chat history text for download."""
    if "messages" not in st.session_state or not st.session_state.messages:
        return ""
    output = io.StringIO()
    start_index = 1 if isinstance(st.session_state.messages[0], SystemMessage) else 0
    for msg in st.session_state.messages[start_index:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        output.write(f"{role}: {msg.content}\n")
    return output.getvalue()

# --- Streamlit UI ---

st.set_page_config(page_title="AFM Defect Assistant (LangGraph)", page_icon="🔬")

st.title("🔬 AFM Image Defect Classification + LLM-based AFM Assistant")
st.write("Upload an AFM image, get a classification, and chat with an AI assistant about the result.")

# --- Sidebar Controls ---
st.sidebar.header("Settings")

# Model Selection
provider = st.sidebar.selectbox("LLM Provider", ["OpenAI", "Anthropic"], key="provider_select")

api_key = None
api_key_name = ""
if provider == "OpenAI":
    default_model = "gpt-4o"
    available_models = ["gpt-4o", "o3-mini"]
    api_key = os.getenv("OPENAI_API_KEY")
    api_key_name = "OPENAI_API_KEY"
elif provider == "Anthropic":
    default_model = "claude-3-5-sonnet-latest"
    available_models = ["claude-3-5-sonnet-latest", "claude-3-7-sonnet-latest"]
    api_key = os.getenv("ANTHROPIC_API_KEY")
    api_key_name = "ANTHROPIC_API_KEY"
else:
    st.sidebar.error("Invalid provider selected.")
    st.stop()

# Display warning if API key is missing
if not api_key:
    st.sidebar.warning(f"{provider} API key not found. Please set the {api_key_name} environment variable.")

model_name = st.sidebar.selectbox(f"Choose {provider} Model", available_models, index=available_models.index(default_model), key="model_name_select")
temperature = st.sidebar.slider("LLM Temperature", 0.0, 1.0, 0.3, 0.05, key="temp_slider")

# Clear Chat Button
if st.sidebar.button("Start New Session", key="clear_chat_button"):
    keys_to_clear = ["messages", "current_label", "uploaded_file_state", "llm", "graph"]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# --- Main Page Logic ---

# File Uploader
uploaded_file = st.file_uploader("Upload an AFM image", type=["jpg", "jpeg", "png"], key="file_uploader")

# Manage state based on uploaded file
if uploaded_file is not None:
    new_file_id = uploaded_file.file_id
    # Check if it's a new file or the same one to avoid re-processing
    if "uploaded_file_state" not in st.session_state or st.session_state.uploaded_file_state["id"] != new_file_id:
        st.session_state.uploaded_file_state = {"id": new_file_id, "name": uploaded_file.name}
        # Clear previous chat/state if a new file is uploaded
        keys_to_reset = ["messages", "current_label", "llm", "graph"]
        for key in keys_to_reset:
            if key in st.session_state:
                del st.session_state[key]

    # --- Image Processing and Classification ---
    try:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption=f"Uploaded: {st.session_state.uploaded_file_state['name']}", width=200)

        model = load_vgg16()
        class_names = load_class_labels()

        with st.spinner("Classifying image..."):
            class_label = predict_image_class(img, model, class_names)
        st.success(f"**Predicted Class Label:** {class_label}")

        # --- LLM and Graph Initialization ---
        label_changed = ("current_label" not in st.session_state or
                         st.session_state.current_label != class_label)

        # Initialize LLM and Graph if not present or if label changed
        if "llm" not in st.session_state or "graph" not in st.session_state or label_changed:
            if not api_key:
                st.error(f"Cannot proceed without {api_key_name}. Please set it in your environment variables.")
                st.stop()
            try:
                # Initialize LLM using the function from graph_logic.py
                st.session_state.llm = initialize_llm(provider, model_name, temperature, api_key)
                # Create the graph using the function from graph_logic.py
                st.session_state.graph = create_chat_graph(st.session_state.llm)
                st.session_state.current_label = class_label

                # Define the system prompt and initial state for the graph
                system_prompt_content = (
                    f"You are an expert in atomic force microscopy (AFM). "
                    f"The user has uploaded an image classified as '{class_label}'. "
                    "Your role is to help the user understand this classification, potential causes, "
                    "and how to potentially avoid or address the issue represented by this classification. "
                    "Provide concise, technically accurate, and helpful answers. Avoid speculation if unsure."
                )
                system_message = SystemMessage(content=system_prompt_content)
                st.session_state.messages = [system_message] # Initialize message history

            except Exception as e:
                st.error(f"Failed to initialize LLM or Graph: {e}")
                st.stop()

        # --- Chat Interface ---
        st.divider()
        st.header(f"Chat about '{st.session_state.current_label}'")

        # Display existing messages (skip system message)
        if "messages" in st.session_state:
            for i, msg in enumerate(st.session_state.messages):
                if i == 0 and isinstance(msg, SystemMessage):
                    continue
                with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
                    st.markdown(msg.content)

        # Chat input
        if prompt := st.chat_input("Ask a question about the detected defect..."):
            # Add user message to state and display it
            st.session_state.messages.append(HumanMessage(content=prompt))
            with st.chat_message("user"):
                st.markdown(prompt)

            # Prepare the input state for the graph
            current_graph_state: GraphState = {"messages": st.session_state.messages}

            # Invoke the graph
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Invoke the graph with the current state
                        response_state = st.session_state.graph.invoke(current_graph_state)

                        # Update session state with the full response history from the graph
                        st.session_state.messages = response_state['messages']
                        ai_response_content = st.session_state.messages[-1].content
                        st.markdown(ai_response_content)

                    except Exception as e:
                        st.error(f"Error during chat generation: {e}")
                        # Roll back user message if AI fails
                        if st.session_state.messages and isinstance(st.session_state.messages[-1], HumanMessage):
                            st.session_state.messages.pop()


        # --- Download Chat Button ---
        if len(st.session_state.get("messages", [])) > 1: # Show only if conversation started
             st.divider()
             chat_text = download_chat_history()
             st.download_button(
                 label="Download Chat History",
                 data=chat_text,
                 file_name=f"afm_chat_{st.session_state.current_label.replace(' ', '_')}.txt",
                 mime="text/plain"
             )

    except Exception as e:
        st.error(f"An error occurred processing the image or during chat setup: {e}")
        if "uploaded_file_state" in st.session_state:
            del st.session_state.uploaded_file_state # Reset if critical error occurs

elif "uploaded_file_state" in st.session_state:
     # If file uploader is cleared by the user after a file was processed
     keys_to_clear = ["messages", "current_label", "uploaded_file_state", "llm", "graph"]
     for key in keys_to_clear:
         if key in st.session_state:
             del st.session_state[key]
     st.rerun()

else:
    st.info("Please upload an image to start the analysis and chat.")

# --- END OF FILE app_main.py ---