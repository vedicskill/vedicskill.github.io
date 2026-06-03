---
sidebar_position: 1
---

# Vision Transformer - High Level Overview

## What is a Vision Transformer?

**Vision Transformer (ViT)** is a revolutionary architecture that applies the Transformer model (originally designed for NLP) directly to image classification tasks by treating images as sequences of patches.

Instead of using convolutional layers, ViT divides images into small patches, embeds them, and processes them through self-attention layers - the same mechanism that powers modern large language models like GPT and BERT.

## The Big Idea

### Traditional Approach (CNN)
```
Image → Convolution Filters → Feature Maps → Classification
        (local processing first)
```

### Vision Transformer Approach
```
Image → Patch Division → Patch Embedding → Self-Attention → Classification
        (global context from the start)
```

**Key Difference:** ViT processes the **entire image globally** from the beginning, not building up from local features like CNNs do.

## Vision Transformer Architecture

### High-Level Pipeline

```
Input Image (224×224×3)
        ↓
1. Patch Splitting (divide into 16×16 patches)
        ↓
196 Patches (14×14 grid)
        ↓
2. Patch Embedding (768-dimensional vectors)
        ↓
3. Add Position Embedding (encode patch positions)
        ↓
4. Add Class Token [CLS] (special token for classification)
        ↓
Sequence of 197 Tokens (196 patches + 1 class token)
        ↓
5. Transformer Encoder (12 layers of self-attention)
        ↓
        ↓
6. Classification Head (maps to 1000 classes)
        ↓
Output Logits
```

## Detailed Components

### 1. Patch Splitting

The image is divided into non-overlapping patches.

```
224×224 Image (14×14 patches)
┌─────┬─────┬─────┬──────┐
│     │     │     │      │ Each patch: 16×16×3
├─────┼─────┼─────┼──────┤
│     │     │     │      │
├─────┼─────┼─────┼──────┤
│     │     │     │      │ Total patches: 14×14 = 196
└─────┴─────┴─────┴──────┘
```

**Calculation:**
- Patch size: 16×16
- Image size: 224×224
- Number of patches: (224/16) × (224/16) = **14 × 14 = 196 patches**
- Each patch has: 16 × 16 × 3 = **768 values**

### 2. Patch Embedding

Each patch is flattened and linearly projected to an embedding vector.

```python
# Each patch: [768] → [768-dimensional embedding]
patch_embedding = Linear(768, embed_dim)  # e.g., embed_dim=768

# All patches embedded
patches: shape (196, 768)
```

The embedding dimensionality:
- **ViT-Base:** 768 dimensions
- **ViT-Large:** 1024 dimensions
- **ViT-Huge:** 1280 dimensions

### 3. Position Embedding

Position information is added to each patch embedding so the model knows where each patch came from.

```
Patch Position Embedding for 196 patches:
- Patch at position (0,0): add position_emb_0
- Patch at position (0,1): add position_emb_1
- Patch at position (13,13): add position_emb_195

Each position has a learnable 768-dimensional vector
```

Two common approaches:
- **Learnable Position Embeddings** - Learned during training
- **Sinusoidal Position Embeddings** - Pre-computed based on position

**After adding position embeddings:**
```
Input to Transformer = Patch Embedding + Position Embedding
```

### 4. Class Token [CLS]

A special learnable token is prepended to the sequence.

```
Sequence before transformer:
[CLS] + patch_0 + patch_1 + ... + patch_195

Total tokens: 197 (1 class token + 196 patch tokens)
```

The [CLS] token:
- Learned from the data
- Aggregates information from all patches through self-attention
- Used for final classification

### 5. Transformer Encoder

The sequence of tokens is processed through multiple transformer encoder layers.

```
Input: [197, 768]  (197 tokens, each 768-dimensional)
        ↓
Transformer Layer 1: Multi-Head Attention + Feed-Forward
        ↓
[197, 768]
        ↓
Transformer Layer 2: Multi-Head Attention + Feed-Forward
        ↓
... (repeat 12 times for ViT-Base)
        ↓
Transformer Layer 12: Multi-Head Attention + Feed-Forward
        ↓
Output: [197, 768]  (processed tokens)
```

Each transformer layer contains:
- **Multi-Head Self-Attention** - Allows patches to attend to each other
- **Feed-Forward Network** - Non-linear transformations
- **Layer Normalization** - Stabilizes training
- **Residual Connections** - Similar to ResNet skip connections

### 6. Classification Head

Only the [CLS] token's output is used for classification.

```
[CLS] token output: [1, 768]
        ↓
Linear Layer: [768] → [1000 classes]
        ↓
Logits: [1, 1000]
        ↓
Softmax → Probabilities
```

## Understanding Self-Attention

Self-attention is the core mechanism of transformers. It answers: **"Which patches should I pay attention to?"**

### Simplified Self-Attention Process

```
For each patch:
1. Query (Q): What am I looking for?
2. Key (K): What do other patches have?
3. Value (V): What's the actual content of other patches?

Attention Score = Softmax(Q · K / √d) · V

Result: Each patch is a weighted combination of all patches!
```

### Visual Example

For a dog image:
- The "head" patch might attend heavily to: other "head" patches, "ear" patches
- The "leg" patch might attend to: other "leg" patches, "body" patches
- The [CLS] token attends to: all patches equally (to understand the whole image)

```
Dog Image:

     ┌─────────────────┐
     │     Ears        │
     ├──────┬──────────┤
     │ Head │  Eyes    │
     ├──────┼──────────┤
     │ Body │          │
     ├──────┼──────────┤
     │Legs  │  Tail    │
     └──────┴──────────┘

Self-Attention Connections:
Head → pays more attention to Eyes, Ears, Body
Ears → pays more attention to Head, Eyes
Legs → pays more attention to Body
[CLS] → distributed attention across all parts
```

## Vision Transformer Variants

Different sized models for different needs:

| Model | Layers | Heads | Dim | Parameters | Speed | Accuracy |
|-------|--------|-------|-----|-----------|-------|----------|
| ViT-Base | 12 | 12 | 768 | 86M | Medium | 77.9% |
| ViT-Large | 24 | 16 | 1024 | 307M | Slow | 82.6% |
| ViT-Huge | 32 | 16 | 1280 | 630M | Very Slow | 84.2% |

More layers, dimensions, and attention heads → Better accuracy but slower and more parameters.

## Code Example: Using Vision Transformer

```python
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image

# Load pre-trained ViT model
model_name = "google/vit-base-patch16-224"
model = ViTForImageClassification.from_pretrained(model_name)
processor = ViTImageProcessor.from_pretrained(model_name)
model.eval()

# Load image
image = Image.open('dog.jpg')

# Preprocess image
inputs = processor(images=image, return_tensors="pt")

# Inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()

# Get class label
print(f"Predicted class: {model.config.id2label[predicted_class_idx]}")
```

## How Patches Are Visualized

For a 224×224 image with 16×16 patches:

```
All 196 Patches (14×14 grid):

┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
└──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
```

Each small box is a 16×16 patch of the original image.

## Key Advantages of Vision Transformers

### 1. **Global Context from Start**
- Unlike CNNs that build up gradually, ViT sees global patterns immediately
- Better for capturing relationships between distant image regions

### 2. **Scalable with Data**
- Performance improves with larger datasets
- Pre-training on huge datasets (JFT-300M) shows major benefits

### 3. **Flexible Architecture**
- Same architecture can be applied to different tasks
- Easier to adapt to new domains

### 4. **Superior Performance on Large Data**
- With sufficient data, ViT outperforms CNNs
- Sets new state-of-the-art records

### 5. **Interpretable Attention**
- Can visualize which patches the model attends to
- Easier to understand what the model is "seeing"

## Limitations of Vision Transformers

### 1. **Computational Cost**
- Self-attention has O(n²) complexity (n = number of patches)
- Slower inference than CNNs
- Requires more computational resources

### 2. **Data Hungry**
- Needs large datasets to perform well
- Pre-training on millions of images is almost essential
- Not ideal when data is limited

### 3. **Positional Encoding**
- Learned position encodings don't generalize well to different image sizes
- Requires interpolation for different input sizes

### 4. **Model Size**
- ViT models tend to have more parameters than CNNs
- Larger memory requirements

## ResNet vs Vision Transformer

### Design Philosophy
| Aspect | ResNet | Vision Transformer |
|--------|--------|-------------------|
| **Building Approach** | Bottom-up (local → global) | Top-down (global from start) |
| **Feature Learning** | Hierarchical convolutions | Self-attention over patches |
| **Inductive Bias** | Strong spatial locality | Minimal inductive bias |
| **Data Requirement** | Works well with limited data | Needs large datasets |
| **Compute** | Efficient | Higher computational cost |
| **Accuracy (Limited Data)** | Better | Worse |
| **Accuracy (Huge Data)** | Good | Better |

### Practical Comparison

```python
# ResNet-50: Good, fast, proven
# - 26M parameters
# - Fast inference
# - Works with 1000s of images

# ViT-Base: Better on large data
# - 86M parameters
# - Slower inference
# - Needs millions of images to shine
```

## When to Use Vision Transformers

✅ **Use ViT when:**
- You have large datasets (millions of images)
- You can afford higher computational cost
- You need state-of-the-art accuracy
- You need interpretability through attention visualization
- You're working with diverse image types

❌ **Avoid ViT when:**
- Data is limited (< 100K images)
- You need fast inference on edge devices
- You have computational constraints
- You need proven stability (ResNet is more established)

## The Future: Combining Both Approaches

Recent research explores:
- **Hybrid Models** - Combining CNN and Transformer strengths
- **Efficient Transformers** - Reducing computational complexity
- **Multi-Scale Transformers** - Processing different image scales
- **Local Attention** - Instead of global attention for efficiency

## Summary

Vision Transformers represent a paradigm shift in computer vision:
- Process images as sequences of patches
- Use self-attention instead of convolutions
- Achieve state-of-the-art results on large datasets
- Provide a unified architecture across vision and language

While they require more data and computation, their performance and flexibility make them increasingly important in modern computer vision.

---

## Next Steps

- **See ViT in Action** → [Vision Transformer Implementation](vit-implementation.md)
- **Compare with ResNet** → [ResNet Overview](../resnet/resnet-overview.md)
- **Explore the Code** → [Notebooks Guide](../notebooks-guide.md)
