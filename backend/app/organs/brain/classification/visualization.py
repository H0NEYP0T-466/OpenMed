import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import torch
import cv2

def plot_training_curves(history, save_dir):
    """Plots training/validation loss and accuracy."""
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'))
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'accuracy_curves.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, save_dir):
    """Plots normalized confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm_normalized, annot=False, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

def plot_class_distribution(train_labels, val_labels, test_labels, class_names, save_dir):
    """Plots class distribution across splits."""
    x = np.arange(len(class_names))
    width = 0.25
    
    train_counts = [train_labels.count(i) for i in range(len(class_names))]
    val_counts = [val_labels.count(i) for i in range(len(class_names))]
    test_counts = [test_labels.count(i) for i in range(len(class_names))]
    
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.bar(x - width, train_counts, width, label='Train')
    ax.bar(x, val_counts, width, label='Val')
    ax.bar(x + width, test_counts, width, label='Test')
    
    ax.set_ylabel('Counts')
    ax.set_title('Class Distribution by Split')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=90)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'class_distribution.png'))
    plt.close()

def plot_sample_predictions(images, true_labels, pred_labels, class_names, save_dir):
    """Plots 4x4 grid of predictions."""
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            # Unnormalize if needed based on transforms, assuming standardization
            img = images[i].numpy().transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)
            
            ax.imshow(img)
            true_cls = class_names[true_labels[i]]
            pred_cls = class_names[pred_labels[i]]
            
            color = 'green' if true_labels[i] == pred_labels[i] else 'red'
            ax.set_title(f"True: {true_cls}\nPred: {pred_cls}", color=color, fontsize=8)
            ax.axis('off')
            
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sample_predictions.png'))
    plt.close()

def plot_gradcam_samples(model, dataset, device, class_names, save_dir, num_samples=8):
    """Plots Grad-CAM overlays."""
    try:
        from .model import GradCAM
    except (ImportError, ValueError):
        from model import GradCAM
    
    for param in model.parameters():
        param.requires_grad = True
        
    target_layer = model.conv_head
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    with GradCAM(model, target_layer) as grad_cam:
        for i, idx in enumerate(indices):
            if i >= 8: break
            
            img_tensor, label, _ = dataset[idx]
            img_input = img_tensor.unsqueeze(0).to(device)
            img_input.requires_grad = True
            
            cam = grad_cam.generate(img_input, label)
            
            img_np = img_tensor.numpy().transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = std * img_np + mean
            img_np = np.clip(img_np, 0, 1)
            
            if len(cam.shape) == 2:
                cam = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
                heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                heatmap = np.float32(heatmap) / 255
                heatmap = heatmap[:, :, ::-1] # BGR to RGB
                overlay = heatmap + np.float32(img_np)
                max_val = np.max(overlay)
                if max_val > 0:
                    overlay = overlay / max_val
            else:
                overlay = img_np
                
            ax = axes[i // 4, i % 4]
            ax.imshow(overlay)
            ax.set_title(f"Class: {class_names[label]}", fontsize=9)
            ax.axis('off')
            
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gradcam_samples.png'))
    plt.close()
    
    for param in model.parameters():
        param.requires_grad = False

def plot_roc_curves(y_true, y_scores, class_names, save_dir):
    """Plots macro-average and per-class ROC curves."""
    n_classes = len(class_names)
    y_true_onehot = np.eye(n_classes)[y_true]
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    # Macro average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr["macro"], tpr["macro"],
             label=f'macro-average ROC curve (area = {roc_auc["macro"]:0.2f})',
             color='navy', linestyle=':', linewidth=4)
             
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, 'roc_curves_macro.png'))
    plt.close()

def plot_pr_curves(y_true, y_scores, class_names, save_dir):
    """Plots PR curves."""
    n_classes = len(class_names)
    y_true_onehot = np.eye(n_classes)[y_true]
    
    precision = dict()
    recall = dict()
    
    plt.figure(figsize=(10, 8))
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_onehot[:, i], y_scores[:, i])
        # plt.plot(recall[i], precision[i], lw=1, alpha=0.3) # Too cluttered
        
    # Compute micro-average PR curve
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_true_onehot.ravel(), y_scores.ravel())
    
    plt.plot(recall["micro"], precision["micro"], color='gold', lw=2,
             label='micro-average Precision-recall curve')
             
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(save_dir, 'pr_curves_micro.png'))
    plt.close()
