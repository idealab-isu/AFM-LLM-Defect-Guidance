import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_Weights

class VGG16Model(nn.Module):
    def __init__(self, num_classes=4):
       super(VGG16Model, self).__init__()
       #self.vgg16 = models.vgg16(pretrained=True)
       # Replace pretrained=True with weights=VGG16_Weights.DEFAULT
       #self.vgg16 = models.vgg16(weights=VGG16_Weights.DEFAULT)
       # Manually load the weights
       self.vgg16 = models.vgg16()
       state_dict = torch.load("/work/mech-ai/angona3/Trial/vgg16_weights.pth", map_location=torch.device("cpu"))
       self.vgg16.load_state_dict(state_dict)
       self.vgg16.classifier[6] = nn.Linear(4096, num_classes) ##In PyTorch CrossEntropyLoss Handles Softmax Internally
       ###Adding an explicit softmax layer would create numerical instability by double-applying softmax (once in your model, once in the loss), 
       # leading to incorrect gradient calculations during training.

       self.vgg16.classifier[2] = nn.Dropout(p=0.6)  # or 0.7 ##introducing dropout for regularization
       self.vgg16.classifier[5] = nn.Dropout(p=0.6)  # or 0.7

       # Freeze all the convolutional layers (feature extractor part)
       # The classifier layers (fully connected layers) remain trainable
       for param in self.vgg16.features.parameters():
           param.requires_grad = False
           #param.requires_grad = True  # Unfreeze the last two fully connected layers
        
         # Unfreeze Conv Block 4 and Conv Block 5 (512 filters, 3x3 filters, same padding) 
        # best: conv_layers_to_unfreeze = [17, 19, 21, 24, 26, 28] 
       conv_layers_to_unfreeze = [17, 19, 21, 24, 26, 28] 
       for layer_idx in conv_layers_to_unfreeze: 
          for param in self.vgg16.features[layer_idx].parameters(): 
               param.requires_grad = True




       # Unfreeze the last two fully connected layers
       # Unfreeze all fully connected layers
       for param in self.vgg16.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.vgg16(x)

#Fine-tuning the entire network can lead to better performance compared to freezing layers 
##because the model can adjust both the feature extractor and classifier to your specific dataset.
##Yes, nn.CrossEntropyLoss in PyTorch is explicitly designed to handle raw logits (ranging from −∞to +∞) directly and efficiently. 
##CrossEntropyLoss=Softmax(logits)+Log+NLLLoss

##### In newer versions of torchvision, the weights argument replaces pretrained. 
# #The VGG16_Weights.DEFAULT is the new way to specify that anyone want the pretrained weights, and it is the preferred method.
##Deprecation Warning Fix: Replace pretrained=True with weights=VGG16_Weights.DEFAULT. 
##Security Warning Fix: You don't need to use torch.load in this case because you're not loading a pre-trained model from a .pth file.
##from torchvision.models import VGG16_Weights
## self.vgg16 = models.vgg16(weights=VGG16_Weights.DEFAULT)


## adding dropout to VGG16
#class VGG16Model(nn.Module):
#    def __init__(self, num_classes=4):
#        super(VGG16Model, self).__init__()
#        self.vgg16 = models.vgg16(pretrained=True)
#        self.vgg16.classifier[6] = nn.Sequential(
#            nn.Linear(4096, 1024),  # First layer reduces dimensions to 1024
#            nn.ReLU(),             # Adds non-linearity to increase learning capacity
#            nn.Dropout(0.5),       # Introduces dropout for regularization
#            nn.Linear(1024, num_classes)  # Second layer maps to the desired number of classes
#        )

#    def forward(self, x):
#        return self.vgg16(x)


#import torch.nn as nn
#from torchvision import models

#def get_resnet18_model(num_classes):
#    model = models.resnet18(pretrained=True)
#    model.fc = nn.Linear(model.fc.in_features, num_classes)
#    return model

##VGG16 structure
#(vgg16.classifier): Sequential(
#  (0): Linear(25088, 4096)
#  (1): ReLU(inplace=True)
#  (2): Dropout(p=0.5, inplace=False)
#  (3): Linear(4096, 4096)
#  (4): ReLU(inplace=True)
#  (5): Dropout(p=0.5, inplace=False)
#  (6): Linear(4096, num_classes)
#)