import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from classification_model.model import VGG16Model
from PIL import Image
import json

from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage


# Load your PyTorch VGG16 model (pre-trained)
@st.cache_resource
def load_vgg16(path='./classification_model/best_model.pth', device='cpu'):
    model = VGG16Model()
    model.load_state_dict(torch.load(path, map_location=device))  # Or GPU if available
    model.eval()
    return model

# Load ImageNet class labels (you need a mapping file)
@st.cache_resource
def load_class_labels():
    class_names = {0:'good_images', 1:'Imaging Artifact', 2:'Not Tracking', 3:'Tip Contamination'}
    return class_names

# Preprocess uploaded image
def preprocess_image(img):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.3718, 0.1738, 0.0571], 
            std=[0.2095, 0.2124, 0.1321]
        ),
    ])
    img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension
    return img_tensor

# Predict using PyTorch model
def predict_image_class(img, model, class_names):
    img_tensor = preprocess_image(img)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        top_prob, top_idx = torch.topk(probs, 1)
        class_label = class_names[top_idx.item()]
    return class_label

# --- Streamlit App ---
st.title("🖼️ Image Classification + LLM Assistant (PyTorch Version)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    model = load_vgg16()
    class_names = load_class_labels()

    class_label = predict_image_class(img, model, class_names)

    st.success(f"**Predicted Class Label:** {class_label}")

    # Initialize or clear chat history if new image uploaded
    if "current_label" not in st.session_state or st.session_state.current_label != class_label:
        st.session_state.current_label = class_label
        st.session_state.messages = [
            SystemMessage(content=f"You are an expert assistant. The detected object in the image is '{class_label}'. Answer user questions based only on this context.")
        ]

    # Show previous chat
    st.header("Chat about the Detected Object")
    for msg in st.session_state.messages[1:]:  # Skip system prompt
        if isinstance(msg, HumanMessage):
            st.markdown(f"**You:** {msg.content}")
        else:
            st.markdown(f"**Assistant:** {msg.content}")

    # Input for user question
    user_question = st.text_input("Ask your question:")

    if user_question:
        # Add user question to chat history
        st.session_state.messages.append(HumanMessage(content=user_question))
        
        # Query LLM
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0.3)
        response = llm(st.session_state.messages)
        
        # Add assistant response to chat history
        st.session_state.messages.append(response)
        
        # Rerun to refresh chat
        st.experimental_rerun()
