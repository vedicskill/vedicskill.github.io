---
sidebar_position: 1
---


# ResNet - High Level Overview

## What is ResNet?

**ResNet** (Residual Network) is a revolutionary deep convolutional neural network architecture that enabled training of very deep networks through the introduction of **skip connections** (also called residual connections).

ResNet won the ImageNet Large Scale Visual Recognition Challenge in 2015 and fundamentally changed how we design deep neural networks.

## The Problem ResNet Solves

Before ResNet, there was a critical problem: **deeper networks performed worse than shallower ones!**

```
Traditional Deep Network Problem:

Shallow Network (20 layers):  Accuracy = 85%
Deep Network (56 layers):     Accuracy = 78%  ← Worse!
```

Why? Two main issues:
1. **Vanishing Gradient Problem** - Gradients become too small to update deep layers during training
2. **Degradation Problem** - Adding more layers actually decreases performance

ResNet solved this with a simple yet powerful idea: **skip connections**.

## The Key Innovation: Skip Connections

### Traditional Layer Block
```
Input → [Conv] → [ReLU] → [Conv] → Output
        (Only learns transformations)
```

### ResNet Block (Skip Connection)
```
        ┌─────────────────────────────┐
        │                             │
Input → [Conv] → [ReLU] → [Conv] → (+) → Output
```

**The Skip Connection:**
- Allows the original input to "skip over" the convolutional layers
- The output becomes: `F(x) + x` (transformation + identity)
- The network learns the **residual** (difference) rather than the whole transformation

### Why This Matters

With skip connections:
- Gradients can flow directly back through the skip path
- Deeper networks don't suffer from degradation
- Networks naturally learn useful identity mappings

## ResNet Architecture

### High-Level Structure

```
Input Image (224×224×3)
        ↓
Convolution Layer (7×7, stride=2)
        ↓
Batch Normalization + ReLU
        ↓
Max Pooling
        ↓
Residual Block Group 1 (Multiple blocks)
        ↓
Residual Block Group 2
        ↓
Residual Block Group 3
        ↓
Residual Block Group 4
        ↓
Average Pooling
        ↓
Fully Connected Layer (1000 classes)
        ↓
Output (Logits)
```

### ResNet Variants

Different depths for different needs:

| Model | Depth | Parameters | Speed | Accuracy |
|-------|-------|-----------|-------|----------|
| ResNet-18 | 18 layers | 11.7M | Very Fast | Good |
| ResNet-34 | 34 layers | 21.8M | Fast | Very Good |
| **ResNet-50** | **50 layers** | **25.6M** | Medium | Excellent |
| ResNet-101 | 101 layers | 44.5M | Slow | Outstanding |
| ResNet-152 | 152 layers | 60.2M | Very Slow | Outstanding |

### ResNet-50 Architecture Details

ResNet-50 is the most commonly used variant. Here's its detailed structure:

```
Stage 0 (Initial Layer):
├─ Conv 7×7, stride=2
├─ Batch Norm
├─ ReLU
└─ Max Pool 3×3, stride=2

Stage 1: 64 filters
├─ 3 Residual Blocks
└─ Output: 56×56×64

Stage 2: 128 filters  
├─ 4 Residual Blocks
└─ Output: 28×28×128

Stage 3: 256 filters
├─ 6 Residual Blocks
└─ Output: 14×14×256

Stage 4: 512 filters
├─ 3 Residual Blocks
└─ Output: 7×7×512

Average Pooling → Flatten → FC (1000) → Output
```

## Two Types of Residual Blocks

### 1. Basic Block (Used in ResNet-18, ResNet-34)

```
Input ─→ Conv (3×3) → BN → ReLU → Conv (3×3) → BN ─→ (+) → ReLU → Output
         └────────────────────────────────────────────┘
                    Skip Connection
```

Simple, used in smaller networks.

### 2. Bottleneck Block (Used in ResNet-50+)

```
Input ─→ Conv(1×1) → BN → ReLU
         → Conv(3×3) → BN → ReLU
         → Conv(1×1) → BN ──→ (+) → ReLU → Output
         └──────────────────────┘
              Skip Connection
```

More efficient through dimensionality reduction:
- 1×1 convolution **reduces** dimensions (cheaper)
- 3×3 convolution operates on reduced dimensions
- 1×1 convolution **expands** dimensions back
- Overall: fewer parameters, same depth

## How ResNet Processes Images

### Step-by-Step Visual Processing

```
Input: 224×224×3 RGB Image
        ↓
[Conv 7×7, stride=2] → Detects large-scale features
112×112×64
        ↓
[Residual Blocks] → Detects edges, textures
56×56×64
        ↓
[More Residual Blocks] → Detects patterns, shapes
28×28×128
        ↓
[More Residual Blocks] → Detects object parts
14×14×256
        ↓
[More Residual Blocks] → Detects high-level concepts
7×7×512
        ↓
[Global Average Pooling] → Summarize features
512-dimensional vector
        ↓
[Fully Connected] → Map to 1000 classes
1000-dimensional logits
        ↓
[Softmax] → Convert to probabilities
Output: Class probabilities
```

### Hierarchical Feature Learning

ResNet learns a hierarchy of features:

**Early Layers (Shallow):**
- Detect simple features: edges, corners, textures
- Low-level, generic to all images

**Middle Layers (Medium Depth):**
- Detect patterns and shapes
- More specific features: wheels, windows

**Deep Layers:**
- Detect high-level concepts: car, dog, person
- Specific to the classification task

## Key Advantages of ResNet

### 1. **Enables Deep Networks**
```python
# Before ResNet: Adding layers decreased performance
# After ResNet: Deeper = Better
ResNet-50 > ResNet-34 > ResNet-18
```

### 2. **Better Gradient Flow**
- Skip connections create "highways" for gradients
- Enables efficient training of 100+ layer networks
- Speeds up convergence

### 3. **Proven and Reliable**
- Widely used in production systems
- Extensive research and optimization
- Robust and stable

### 4. **Computational Efficiency**
- Good accuracy-to-parameter ratio
- Fast inference compared to larger networks
- Suitable for real-time applications

### 5. **Transfer Learning Friendly**
- Pre-trained weights available
- Works well when fine-tuned on new tasks
- Rapid prototyping for custom tasks

## Code Example: Using ResNet-50

```python
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL import Image

# Load pre-trained ResNet-50
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()  # Set to evaluation mode

# Prepare image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load and preprocess image
image = Image.open('dog.jpg')
image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Inference
with torch.no_grad():
    logits = model(image_tensor)
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)

# Get class label
categories = weights.meta["categories"]
print(f"Predicted: {categories[predicted_class.item()]}")
print(f"Confidence: {probabilities.max().item():.2%}")
```

## ResNet vs Vision Transformer

| Aspect | ResNet | Vision Transformer |
|--------|--------|-------------------|
| **Approach** | Convolutional | Patch + Attention |
| **Local/Global** | Local features first | Global context |
| **Parameters** | Smaller | Usually larger |
| **Speed** | Very fast | Slower |
| **Efficiency** | Less data needed | Needs more data |
| **Interpretability** | Somewhat interpretable | More complex |
| **SOTA** | Very good | Better on large datasets |

## When to Use ResNet

✅ **Use ResNet when:**
- You need fast inference
- You have limited computational resources
- You need proven, stable performance
- Data is limited (transfer learning)
- You need interpretability

❌ **Avoid ResNet when:**
- You have access to very large datasets
- You can afford higher computational cost
- You need the absolute best accuracy
- You need global context awareness

## Limitations of ResNet

1. **Local-first Processing** - Processes local regions first, slower to develop global understanding
2. **Less Flexible** - Architecture is relatively fixed; harder to modify for different tasks
3. **Limited Long-range Dependencies** - Not ideal for capturing relationships between distant image regions
4. **Inductive Bias** - Convolutions assume spatial locality; some tasks need less local bias

## Summary

ResNet revolutionized deep learning by:
- Introducing skip connections to overcome vanishing gradients
- Enabling training of very deep networks
- Providing proven, fast, and reliable image classification
- Becoming the foundation for countless computer vision systems

Its success led to widespread adoption across industries and paved the way for future innovations like Vision Transformers.

---

## Next Steps

- **Learn about Vision Transformers** → [Vision Transformer Overview](../vit/transformer-overview.md)
- **See ResNet in Action** → [ResNet Implementation](resnet-implementation.md)
- **Understand the Code** → [Notebooks Guide](../notebooks-guide.md)
