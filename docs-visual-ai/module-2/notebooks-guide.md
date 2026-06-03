---
sidebar_position: 5
---

# Notebooks Guide

This section provides detailed walkthroughs of the two main Jupyter notebooks that form the practical core of this course.

## Overview

| Notebook | Focus | Key Concepts | Duration |
|----------|-------|--------------|----------|
| **1-understanding-image-classification.ipynb** | ResNet & CNN | Fundamentals, Pre-trained Models | 30 mins |
| **2-vit-teach.ipynb** | Vision Transformer | Patches, Attention, Modern Approach | 40 mins |

## Notebook 1: Understanding Image Classification

**File:** `notebooks/1-understanding-image-classification.ipynb`

### Overview
This notebook introduces image classification using Convolutional Neural Networks through ResNet-50, the most popular pre-trained model.

### What You'll Learn
✅ Load and visualize images  
✅ Understand image data representation  
✅ Use pre-trained ResNet models  
✅ Perform inference on images  
✅ Interpret model outputs  

### Section-by-Section Breakdown

#### Section 1: Import Libraries
```python
import numpy as np
import cv2
from PIL import Image
import torch 
import torchvision
import matplotlib.pyplot as plt
```

**What's happening:**
- NumPy: Array operations
- OpenCV: Image processing
- PIL: Image loading and manipulation
- PyTorch: Deep learning framework
- TorchVision: Computer vision models and utilities
- Matplotlib: Visualization

**Learning point:** These are the essential libraries for image classification tasks.

#### Section 2: Check Versions
```python
print("OpenCV version:", cv2.__version__)
print("Torch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
```

**Why this matters:** Version compatibility is crucial. The code is tested on:
- PyTorch 2.0+
- TorchVision 0.15+
- OpenCV 4.8+

**Your task:** Note the versions - they help with troubleshooting.

#### Section 3: Load and Visualize Image
```python
image = Image.open('../data/image-3.jpg')

plt.imshow(image)
plt.axis('off')
plt.title('Input Image')
plt.show()
```

**What's happening:**
- Load image using PIL
- Display using matplotlib
- `axis('off')` removes axis labels

**Key insight:** The image we'll classify - a basic representation to the model.

#### Section 4: Understand Image Data
```python
img_array = np.asarray(image)
print("Shape ", img_array.shape)
# Output: (height, width, channels)
```

**Understanding the output:**
```
Shape: (480, 640, 3)
├─ 480: Height (vertical pixels)
├─ 640: Width (horizontal pixels)
└─ 3: Channels (R, G, B)

Each pixel has 3 values:
[R_value, G_value, B_value]
```

**Key concept:** Images are just 3D arrays of numbers!

#### Section 5: Load Pre-trained ResNet
```python
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
import torch.nn.functional as F

# Define preprocessing
transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# Load model
weights = ResNet50_Weights.DEFAULT
model = resnet50(pretrained=True)
model.eval()
```

**Breaking this down:**

1. **Import weights:** `ResNet50_Weights.DEFAULT` includes:
   - The pre-trained weights (trained on ImageNet)
   - Metadata (1000 class labels)

2. **Define transforms:** Preprocessing pipeline
   - Resize to 224×224 (standard for ResNet)
   - Convert to tensor (0-1 range)

3. **Load model:** Create model with pre-trained weights
   - `resnet50(pretrained=True)` loads model + weights
   - `model.eval()` sets to evaluation mode (no dropout, batch norm freezes)

**Key point:** Pre-trained weights are crucial - we're using ImageNet knowledge!

#### Section 6: Preprocess Image
```python
image_tensor = transforms(image)
print("After transform shape:", image_tensor.shape)
# Output: torch.Size([3, 224, 224])

# Add batch dimension
input_tensor = image_tensor.unsqueeze(0)
print("With batch dimension:", input_tensor.shape)
# Output: torch.Size([1, 3, 224, 224])
```

**What's happening:**
```
Original image:
[480, 640, 3]
        ↓ (Resize)
[224, 224, 3]
        ↓ (ToTensor)
[3, 224, 224]  (channels first, 0-1 range)
        ↓ (unsqueeze)
[1, 3, 224, 224]  (batch of 1)
```

**Why the batch dimension?**
- Models expect batches
- Even single image needs batch dim
- `unsqueeze(0)` adds dimension at position 0

#### Section 7: Run Inference
```python
with torch.no_grad():
    output = model(input_tensor)

print("Output shape =", output.shape)
# Output: torch.Size([1, 1000])
```

**Understanding the output:**
```
Output shape: [1, 1000]
├─ 1: Batch size (1 image)
└─ 1000: Number of ImageNet classes

Each value is a "logit" (raw, unnormalized score)
Output[0][i] = confidence for class i
```

**Why `torch.no_grad()`?**
- Disables gradient computation
- Faster inference
- Lower memory usage
- Not needed for just prediction

#### Section 8: Convert to Probabilities
```python
probabilities = F.softmax(output, dim=1)
print("Probabilities sum =", probabilities.sum().item())
# Output: 1.0 (normalized!)

# Get predicted class
predicted_idx = torch.argmax(probabilities, dim=1)
print("Predicted class index:", predicted_idx.item())

# Get confidence
confidence = probabilities[0, predicted_idx.item()].item()
print(f"Confidence: {confidence:.2%}")
```

**Understanding softmax:**
```
Logits:        [-2.5, 1.2, 3.8, 0.5, ...]
                    ↓ (softmax)
Probabilities: [0.01, 0.05, 0.90, 0.02, ...]
               └─ Sum = 1.0 (valid probability distribution)
```

**Key insight:** Softmax converts raw scores to probabilities!

#### Section 9: Get Class Label
```python
labels = weights.meta['categories']
print("Categories size =", len(labels))
# Output: 1000

print(f"Predicted: {labels[predicted_idx.item()]}")
print(f"Confidence: {confidence:.2%}")
```

**What's `weights.meta`?**
- Contains metadata from pre-trained weights
- `categories`: List of 1000 ImageNet class names
- `categories[i]` = name of class i

**Complete output:**
```
Predicted: golden_retriever
Confidence: 92.47%
```

### Hands-on Exercises

**Exercise 1: Try Different Images**
```python
# Modify the image path
image = Image.open('../data/image-1.jpg')  # Try different image
# Re-run sections 4-9
```

**Exercise 2: Get Top-5 Predictions**
```python
# Get top 5 probabilities
top5_probs, top5_indices = torch.topk(probabilities, 5, dim=1)

# Display
for prob, idx in zip(top5_probs[0], top5_indices[0]):
    print(f"{labels[idx]}: {prob:.2%}")
```

**Exercise 3: Different ResNet Variants**
```python
from torchvision.models import resnet18, resnet34, resnet50, resnet101

# Try different models
models_to_try = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101
}

# Compare predictions and inference time
```

---

## Notebook 2: Vision Transformer

**File:** `notebooks/2-vit-teach.ipynb`

### Overview
This notebook explores Vision Transformers from first principles, showing how patch-based approaches differ from CNNs.

### What You'll Learn
✅ Understand image patching  
✅ Visualize patches  
✅ Learn about embeddings  
✅ Use Vision Transformer models  
✅ Compare CNN vs Transformer  

### Section-by-Section Breakdown

#### Section 1: Load Image
```python
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt

image = Image.open('../data/image-3.jpg')

plt.imshow(image)
plt.title('Input Image')
plt.axis('off')
plt.show()
```

**Same as Notebook 1** - Load the image we'll work with.

#### Section 2: Manual Patch Creation
```python
# Convert to numpy array
image_array = np.array(image_resize)
print(image_array.shape)

# Split into patches manually
patch_size = 16
patches = []

for y in range(0, 224, patch_size):
    for x in range(0, 224, patch_size):
        patch = image_array[
            y:y+patch_size,
            x:x+patch_size
        ]
        patches.append(patch)

print("Total Patches:", len(patches))
# Output: 196 (14 × 14)
```

**Understanding patching:**
```
224×224 image divided into 16×16 patches:
Number of patches = (224/16) × (224/16) = 14 × 14 = 196

Each patch:
Shape: [16, 16, 3]
Contains: 16×16×3 = 768 values
```

**Visual representation:**
```
┌────┬────┬────┬────┬────┐
│ P0 │ P1 │ P2 │ P3 │... │  Row 0
├────┼────┼────┼────┼────┤
│ P14│ P15│ P16│ P17│... │  Row 1
├────┼────┼────┼────┼────┤
│... │... │... │... │... │
└────┴────┴────┴────┴────┘

Total: 14 rows × 14 cols = 196 patches
```

#### Section 3: Visualize All Patches
```python
fig, axes = plt.subplots(14, 14, figsize=(8,8))

for i, ax in enumerate(axes.flat):
    ax.imshow(patches[i])
    ax.axis("off")
    ax.set_title(str(i), fontsize=6)

plt.suptitle("All 196 Vision Transformer Patches", fontsize=20)
plt.show()
```

**What you're seeing:**
- All 196 patches displayed in a 14×14 grid
- Each small square is a 16×16 pixel patch
- You can see how ViT "sees" the image

**Key insight:** The model processes these patches, not the whole image!

#### Section 4: Flatten Patches
```python
flattened_patches = []

for patch in patches:
    flattened = patch.flatten()
    flattened_patches.append(flattened)

flattened_patches = np.array(flattened_patches)
print(flattened_patches.shape)
# Output: (196, 768)

# Interpretation:
# 196: Number of patches
# 768: Flattened size (16×16×3)
```

**What's happening:**
```
Patch shape: [16, 16, 3]
             ↓ (flatten)
Flattened: [768]

All patches:
[196, 768]
├─ 196 patches
└─ Each patch is a 768-dimensional vector
```

#### Section 5: Efficient Patch Creation
```python
from torch.nn import Unfold

# Create patches using PyTorch
image_tensor = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])(image)

# Add batch dimension
image_tensor = image_tensor.unsqueeze(0)
print(image_tensor.shape)  # [1, 3, 224, 224]

# Create patches
patch_size = 16
unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
patches = unfold(image_tensor)
print("Patches shape:", patches.shape)  # [1, 768, 196]

# Rearrange to [batch, num_patches, patch_dim]
patches = patches.transpose(1, 2)
print("After transpose:", patches.shape)  # [1, 196, 768]
```

**Understanding Unfold:**
- `kernel_size=16`: Patch size
- `stride=16`: Non-overlapping patches
- Creates sliding windows across image
- Returns shape: [batch, channels×patch_h×patch_w, num_patches]

**After transpose:**
```
From: [1, 768, 196]
To:   [1, 196, 768]

Better format for transformer input
```

#### Section 6: Load Vision Transformer
```python
from transformers import ViTForImageClassification, ViTImageProcessor

model_name = "google/vit-base-patch16-224"
model = ViTForImageClassification.from_pretrained(model_name)
processor = ViTImageProcessor.from_pretrained(model_name)

model.eval()
print(model)
```

**Components:**
1. **Model:** `ViTForImageClassification`
   - Vision Transformer for classification
   - Pre-trained on ImageNet
   - 86M parameters

2. **Processor:** `ViTImageProcessor`
   - Handles image preprocessing
   - Resize, normalize, convert to tensor
   - Consistent with training

#### Section 7: Inference with ViT
```python
# Preprocess image
inputs = processor(images=image, return_tensors="pt")

# Inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Get prediction
predicted_class_idx = logits.argmax(-1).item()
class_label = model.config.id2label[predicted_class_idx]

print(f"Predicted: {class_label}")
```

**Process:**
```
Image → Processor → Preprocessed inputs
                      ↓
                   Model → Logits
                      ↓
                   Softmax → Probabilities
                      ↓
                   argmax → Class index
                      ↓
                   id2label → Class name
```

#### Section 8: Explore Embeddings
```python
# Access embedding layer
embedding_layer = model.vit.embeddings.patch_embeddings

print("Patch Embedding Layer:")
print(f"Input dimension: 768 (16×16×3)")
print(f"Output dimension: 768")

# The embedding projects 768D patch → 768D vector
# This creates learnable embeddings from raw patches
```

**Understanding embeddings:**
```
Raw patch [768] → Linear([768, 768]) → Embedded patch [768]

Each patch gets transformed to a learned representation
```

### Hands-on Exercises

**Exercise 1: Visualize Patches**
```python
# Modify patch size
for patch_size in [8, 16, 32]:
    # Recalculate patches
    num_patches = (224 // patch_size) ** 2
    print(f"Patch size {patch_size}: {num_patches} patches")
    # Display patches
```

**Exercise 2: Compare Predictions**
```python
# ResNet prediction
resnet_pred = resnet_model(input_tensor)
resnet_class = labels[resnet_pred.argmax()]

# ViT prediction
vit_pred = vit_model(**vit_inputs)
vit_class = model.config.id2label[vit_pred.logits.argmax().item()]

print(f"ResNet: {resnet_class}")
print(f"ViT: {vit_class}")
```

**Exercise 3: Visualization**
```python
# Create side-by-side comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Original image with grid
ax = axes[0]
ax.imshow(image)

# Draw patch grid
for i in range(0, 224, 16):
    ax.axhline(y=i, color='white', linewidth=0.5)
    ax.axvline(x=i, color='white', linewidth=0.5)

ax.set_title("Patches")
ax.axis('off')

# Right: Patch visualization
ax = axes[1]
ax.text(0.5, 0.5, "Vision Transformer\nProcesses patches", 
        ha='center', va='center', fontsize=14)
ax.set_title("Processing")

plt.tight_layout()
plt.show()
```

### Comparison: CNN vs Transformer

**Processing Flow Comparison:**

**ResNet (CNN):**
```
224×224 Image
    ↓
7×7 Conv (stride 2) → 112×112×64 features
    ↓
Residual Blocks → hierarchical features
    ↓
14×14×256 features
    ↓
Global Average Pool → 256D vector
    ↓
1000 classes
```

**ViT (Transformer):**
```
224×224 Image
    ↓
Patch Embedding → 196×768 patches
    ↓
Position Embedding → encode positions
    ↓
Transformer Encoder → self-attention
    ↓
[CLS] token output → 768D vector
    ↓
Classification Head → 1000 classes
```

## Summary of Both Notebooks

| Aspect | Notebook 1 (ResNet) | Notebook 2 (ViT) |
|--------|-------------------|------------------|
| **Architecture** | CNN with residual connections | Patch + Transformer |
| **Processing** | Hierarchical, local-first | Global attention |
| **Complexity** | Simpler to understand | More abstract |
| **Speed** | Faster | Slower |
| **Accuracy** | Good | Better on large data |
| **Patches** | Implicit (filters) | Explicit (196 patches) |

## Running Both Notebooks

### Recommended Workflow

**Day 1:**
- Run Notebook 1 (30 minutes)
- Understand ResNet basics
- Experiment with different images

**Day 2:**
- Run Notebook 2 (40 minutes)
- Learn Vision Transformer concepts
- Visualize patches
- Compare predictions

**Day 3:**
- Run both together
- Compare side-by-side
- Answer review questions

## Common Issues and Solutions

### Issue: Image Not Found
```python
# Make sure you're in the right directory
import os
os.chdir('notebooks')  # If needed
image = Image.open('../data/image-3.jpg')
```

### Issue: Model Download Slow
```python
# First download may take a few minutes
# Check internet connection
# Models are cached after first download
```

### Issue: CUDA Memory Error
```python
# Use CPU instead
device = torch.device('cpu')
model = model.to(device)
input_tensor = input_tensor.to(device)
```

---

## Next Steps

- **Explore ResNet Implementation Details** → [ResNet Implementation](resnet/resnet-implementation.md)
- **Explore ViT Implementation Details** → [Vision Transformer Implementation](vit/vit-implementation.md)
- **Review Architecture** → [ResNet Overview](resnet/resnet-overview.md)
