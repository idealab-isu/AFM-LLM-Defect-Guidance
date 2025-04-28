import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from model import VGG16Model
# Load your PyTorch VGG16 model (pre-trained)

def load_vgg16():
    model = VGG16Model()
    model.load_state_dict(torch.load("/work/mech-ai/jrrade/AFM/AFM-LLM-Defect-Guidance/classification_model/best_model.pth", map_location=torch.device('cpu')))  # Or GPU if available
    model.eval()
    return model

# Load ImageNet class labels (you need a mapping file)

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
        _, preds = torch.max(outputs, dim=1)
        print(_, preds)
        probs = F.softmax(outputs, dim=1)
        top_prob, top_idx = torch.topk(probs, 1)
        print(top_prob, top_idx)
        class_label = class_names[top_idx.item()]
    return class_label

img_path = '/work/mech-ai/angona3/Trial/image/Not_Tracking/Not_Tracking_21.jpg'
# img_path = '/work/mech-ai/angona3/Trial/image/Tip_Contamination/Tip_Contamination_17.jpg'
img = Image.open(img_path)
model = load_vgg16()
class_names = load_class_labels()
class_label = predict_image_class(img, model, class_names)
print(class_label)