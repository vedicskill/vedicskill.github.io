---
sidebar_position: 2
---

# Introduction to Image Classification

## What is Image Classification?

**Image Classification** is the task of assigning a single label or category to an entire image. It's one of the most fundamental problems in computer vision and forms the foundation for more complex vision tasks.

### Simple Definition

Image Classification answers the question: **"What is in this image?"**

Given an input image, an image classification model predicts which category from a predefined set of classes the image belongs to.

### Real-World Examples

| Input Image | Predicted Category |
|---|---|
| 🐕 Dog photo | "Dog" |
| 🐱 Cat photo | "Cat" |
| 🚗 Car photo | "Car" |
| ✈️ Airplane photo | "Airplane" |
| 🐴 Horse photo | "Horse" |

## How Does It Work?

### The Classification Pipeline

```
Input Image → Preprocessing → Model → Logits → Probabilities → Class Label
```

**Step-by-Step Process:**

1. **Input Image** - Raw image (JPG, PNG, etc.)
2. **Preprocessing** - Resize, normalize, convert to tensors
3. **Model** - Deep neural network processes the image
4. **Logits** - Raw model outputs (unnormalized scores)
5. **Probabilities** - Convert logits to probabilities using softmax
6. **Prediction** - Select class with highest probability

### Example: Understanding Terminology

```python
# Image comes in
image = load_image('dog.jpg')  # Shape: (480, 640, 3)

# Preprocessing
image = resize(image, (224, 224))  # Standard size
image = normalize(image)           # Normalize pixel values
image = to_tensor(image)           # Convert to tensor

# Model inference
logits = model(image)              # Raw outputs: shape (1000,)

# Convert to probabilities
probabilities = softmax(logits)    # Sum = 1.0

# Get prediction
predicted_class = argmax(probabilities)  # Index of max probability
confidence = probabilities[predicted_class]  # 0.95 (95% confident)

# Output
print(f"Prediction: {class_names[predicted_class]}")
print(f"Confidence: {confidence:.2%}")
```

## Why Image Classification Matters

### Applications

- **Medical Imaging** - Detecting diseases in X-rays and MRI scans
- **Autonomous Vehicles** - Identifying traffic signs and obstacles
- **E-Commerce** - Product categorization and search
- **Social Media** - Content moderation and tagging
- **Agriculture** - Crop disease detection
- **Manufacturing** - Quality control and defect detection

### Foundation for Advanced Tasks

Image classification is the building block for:
- **Object Detection** - Finding and localizing objects in images
- **Semantic Segmentation** - Classifying every pixel in an image
- **Instance Segmentation** - Identifying individual object instances
- **Image Captioning** - Generating descriptions of images
- **Visual Question Answering** - Answering questions about images

## Key Concepts

### Classes and Categories

A classification model is trained on a specific set of classes (categories) it can recognize.

Example: ImageNet with 1000 classes
- Animal classes (dog, cat, bird, etc.)
- Vehicle classes (car, truck, airplane, etc.)
- Furniture classes (chair, table, sofa, etc.)
- And 997 more...

### Confidence Scores

The model doesn't just give a yes/no answer - it provides probabilities:

```
Input: Dog image

Output Probabilities:
- Dog:        95.2%  ← Highest confidence
- Wolf:       3.1%
- Coyote:     1.3%
- Fox:        0.4%
```

### Transfer Learning

Modern approaches use **pre-trained models** trained on millions of images, then fine-tuned for specific tasks. This is far more efficient than training from scratch.

## Two Approaches Covered in This Guide

### 1. CNN-based Approach (ResNet)

**Convolutional Neural Networks** use:
- Local filters that detect features (edges, textures, shapes)
- Hierarchical feature extraction (simple features → complex features)
- Spatial relationships are preserved

**Strengths:**
- Well-established and proven
- Computationally efficient
- Good inductive biases for images
- Excellent performance on many tasks

### 2. Transformer-based Approach (Vision Transformers)

**Vision Transformers** use:
- Self-attention mechanism to see global image context
- Patch-based processing (divide image into patches)
- No convolutional operations needed

**Strengths:**
- Can capture long-range dependencies
- Scales better to large datasets
- More flexible architecture
- State-of-the-art performance on many benchmarks

## Learning Path

This documentation guides you through:

1. **Understand Basics** - Image classification fundamentals
2. **Learn Architecture 1** - ResNet and CNNs
3. **Learn Architecture 2** - Vision Transformers
4. **Understand Data** - Image processing and preprocessing
5. **Explore Code** - Complete implementation walkthrough

Each section builds on the previous ones, creating a comprehensive understanding of image classification.

## About This Guide

This guide uses:
- **PyTorch** for deep learning
- **TorchVision** for pre-trained models
- **Hugging Face** for transformer models
- **OpenCV & Pillow** for image processing

All code examples are practical, runnable, and follow best practices.

---

## Next Steps

Ready to start? Lets dive straight into [ResNet Overview](resnet/resnet-overview.md).

**Common Questions?** Visit our [FAQs](/faqs) page.
