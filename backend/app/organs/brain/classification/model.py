import torch
import torch.nn as nn
import timm
import cv2
import numpy as np

def create_model(num_classes=42, pretrained=True):
    """Creates EfficientNetV2-B2 model from timm."""
    model = timm.create_model('tf_efficientnetv2_b2.in1k', pretrained=pretrained, num_classes=num_classes)
    return model

def get_loss_function(class_weights=None, device='cpu'):
    """Returns CrossEntropyLoss with optional weights."""
    if class_weights is not None:
        return nn.CrossEntropyLoss(weight=class_weights.to(device))
    return nn.CrossEntropyLoss()

def get_optimizer(model, lr=1e-3, weight_decay=1e-4):
    """Returns AdamW optimizer."""
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

def get_scheduler(optimizer, num_epochs, warmup_epochs=5, steps_per_epoch=1):
    """Returns OneCycleLR scheduler."""
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=optimizer.param_groups[0]['lr'],
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch
    )

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handles = [
            target_layer.register_forward_hook(self.save_activation),
            target_layer.register_full_backward_hook(self.save_gradient)
        ]
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        
    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad()
        
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        target = output[0][target_class]
        target.backward()
        
        # Global average pooling on the gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = torch.relu(cam) # ReLU on CAM
        
        cam = cam.cpu().detach().numpy()
        cam = cam - np.min(cam)
        cam_max = np.max(cam)
        if cam_max != 0:
            cam = cam / cam_max

        self.remove_hooks()
        return cam
