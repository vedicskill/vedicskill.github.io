---
sidebar_position: 2
---

# ResNet Implementation Guide

## Complete ResNet Pipeline

This guide walks through implementing image classification with ResNet from start to finish.

## Step 1: Imports and Setup

```python
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
```

**Each import's purpose:**
- `torch.nn`: Neural network layers
- `ResNet50_Weights`: Pre-trained weights
- `transforms`: Image preprocessing
- `PIL.Image`: Image loading
- `matplotlib.pyplot`: Visualization

## Step 2: Load Pre-trained Model

```python
# Load weights
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)

# Set to evaluation mode
model.eval()

# Get metadata
categories = weights.meta['categories']
num_classes = len(categories)

print(f"Model loaded: {model.__class__.__name__}")
print(f"Number of classes: {num_classes}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**Output:**
```
Model loaded: ResNet
Number of classes: 1000
Model parameters: 25,557,032
```

**Key points:**
- `ResNet50_Weights.DEFAULT`: Latest recommended weights
- `model.eval()`: Critical for inference (disables dropout, freezes batch norm)
- `weights.meta`: Contains important metadata

## Step 3: Create Preprocessing Pipeline

```python
# Define transforms
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to 256
    transforms.CenterCrop((224, 224)),  # Crop center 224×224
    transforms.ToTensor(),  # Convert to tensor [0, 1]
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean (per channel)
        std=[0.229, 0.224, 0.225]  # ImageNet std (per channel)
    )
])

print("Preprocessing pipeline created:")
print("1. Resize to 256×256")
print("2. Center crop to 224×224")
print("3. Convert to tensor")
print("4. Normalize with ImageNet statistics")
```

**Why this pipeline?**
- **Resize to 256:** Maintains aspect ratio without distortion
- **CenterCrop to 224:** Standard ResNet input size
- **ToTensor:** Converts PIL Image to torch tensor
- **Normalize:** Uses ImageNet training statistics for compatibility

## Step 4: Load and Preprocess Image

```python
# Load image
image_path = '../data/image-3.jpg'
image = Image.open(image_path).convert('RGB')

print(f"Original image size: {image.size}")

# Apply preprocessing
image_tensor = preprocess(image)

print(f"Tensor shape: {image_tensor.shape}")
print(f"Value range: [{image_tensor.min():.2f}, {image_tensor.max():.2f}]")

# Add batch dimension
batch = image_tensor.unsqueeze(0)
print(f"Batch shape: {batch.shape}")

# Move to device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch = batch.to(device)
model = model.to(device)
print(f"Using device: {device}")
```

**Output example:**
```
Original image size: (640, 480)
Tensor shape: torch.Size([3, 224, 224])
Value range: [-2.03, 2.62]
Batch shape: torch.Size([1, 3, 224, 224])
Using device: cuda
```

## Step 5: Run Inference

```python
# Disable gradient computation
with torch.no_grad():
    # Forward pass
    logits = model(batch)

print(f"Output logits shape: {logits.shape}")
# Output: torch.Size([1, 1000])
# 1 = batch size, 1000 = number of classes

# Examine raw logits
print(f"Min logit: {logits.min():.2f}")
print(f"Max logit: {logits.max():.2f}")
print(f"Mean logit: {logits.mean():.2f}")
```

**Understanding logits:**
```
Logits are raw, unnormalized scores
- Can be negative or very large
- Not probabilities yet
- Need softmax to convert to probabilities
```

## Step 6: Convert Logits to Probabilities

```python
# Apply softmax to convert to probabilities
probabilities = F.softmax(logits, dim=1)

print(f"Probabilities shape: {probabilities.shape}")
print(f"Sum of probabilities: {probabilities.sum(dim=1).item():.4f}")  # Should be ~1.0
print(f"Min probability: {probabilities.min():.4f}")
print(f"Max probability: {probabilities.max():.4f}")
```

**Output:**
```
Probabilities shape: torch.Size([1, 1000])
Sum of probabilities: 1.0000  ← Valid probability distribution!
Min probability: 0.0001
Max probability: 0.9247
```

**What softmax does:**
```
Logits: [-2.5, 1.2, 3.8, 0.5, -1.1]
            ↓ (softmax)
Probs:  [0.001, 0.017, 0.925, 0.050, 0.007]
```

## Step 7: Get Predictions

```python
# Get predicted class
predicted_idx = torch.argmax(probabilities, dim=1).item()
confidence = probabilities[0, predicted_idx].item()

# Get class name
predicted_class = categories[predicted_idx]

print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence:.2%}")
```

**Output example:**
```
Predicted class: golden_retriever
Confidence: 92.47%
```

## Step 8: Get Top-K Predictions

```python
# Get top-5 predictions
top_k = 5
top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)

print(f"\nTop-{top_k} Predictions:")
print("-" * 40)

for i, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0]), 1):
    class_name = categories[idx.item()]
    confidence = prob.item()
    print(f"{i}. {class_name:<30} {confidence:>6.2%}")

print("-" * 40)
```

**Output example:**
```
Top-5 Predictions:
----------------------------------------
1. golden_retriever         92.47%
2. Labrador_retriever       4.32%
3. Yellow_Labrador          2.15%
4. Irish_setter             0.89%
5. Great_Pyrenees           0.17%
----------------------------------------
```

## Step 9: Visualization

```python
def visualize_predictions(image_path, top_k=5):
    """Visualize image with top-k predictions"""
    
    # Load and process image
    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    
    # Get predictions
    with torch.no_grad():
        logits = model(image_tensor)
    
    probabilities = F.softmax(logits, dim=1)
    top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Original image
    ax1.imshow(image)
    ax1.set_title('Input Image')
    ax1.axis('off')
    
    # Right: Top predictions
    class_names = [categories[idx.item()] for idx in top_indices[0]]
    probs = [prob.item() for prob in top_probs[0]]
    
    # Replace underscores and capitalize
    class_names = [name.replace('_', ' ').title() for name in class_names]
    
    bars = ax2.barh(range(top_k), probs)
    ax2.set_yticks(range(top_k))
    ax2.set_yticklabels(class_names)
    ax2.set_xlabel('Confidence')
    ax2.set_title(f'Top-{top_k} Predictions')
    ax2.invert_yaxis()
    
    # Color bars by confidence
    colors = plt.cm.RdYlGn(np.array(probs))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Add percentage labels
    for i, (prob, bar) in enumerate(zip(probs, bars)):
        ax2.text(prob, i, f' {prob:.1%}', va='center')
    
    plt.tight_layout()
    plt.show()
    
    return class_names[0], probs[0]

# Usage
predicted_class, confidence = visualize_predictions('../data/image-3.jpg')
print(f"\nFinal Prediction: {predicted_class} ({confidence:.2%})")
```

## Complete Example Function

```python
class ResNetClassifier:
    """Reusable ResNet classifier"""
    
    def __init__(self, device=None):
        # Load model
        weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=weights)
        self.model.eval()
        self.categories = weights.meta['categories']
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model = self.model.to(self.device)
        
        # Preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def predict(self, image_path, top_k=5):
        """
        Predict class of image
        
        Args:
            image_path: Path to image file
            top_k: Return top-k predictions
            
        Returns:
            dict with prediction results
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(image_tensor)
        
        # Get probabilities
        probabilities = F.softmax(logits, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
        
        # Format results
        results = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            class_idx = idx.item()
            class_name = self.categories[class_idx]
            confidence = prob.item()
            results.append({
                'class': class_name,
                'confidence': confidence,
                'class_idx': class_idx
            })
        
        return {
            'prediction': results[0]['class'],
            'confidence': results[0]['confidence'],
            'top_k': results,
            'image_path': image_path
        }

# Usage
classifier = ResNetClassifier()
results = classifier.predict('../data/image-3.jpg', top_k=5)

print(f"Prediction: {results['prediction']}")
print(f"Confidence: {results['confidence']:.2%}")
print("\nTop-5:")
for i, result in enumerate(results['top_k'], 1):
    print(f"{i}. {result['class']}: {result['confidence']:.2%}")
```

## Batch Processing

```python
def batch_predict(image_paths, top_k=1):
    """Predict on multiple images efficiently"""
    
    classifier = ResNetClassifier()
    results = []
    
    for image_path in image_paths:
        result = classifier.predict(image_path, top_k=top_k)
        results.append(result)
        print(f"✓ {image_path}: {result['prediction']}")
    
    return results

# Usage
images = [
    '../data/image-1.jpg',
    '../data/image-2.jpg',
    '../data/image-3.jpg'
]

batch_results = batch_predict(images)
```

## Performance Tips

### GPU Acceleration
```python
# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Move model to GPU
model = model.to(device)

# For large batches, process in chunks
batch_size = 32
for i in range(0, len(images), batch_size):
    batch = images[i:i+batch_size]
    # Process batch
```

### Model Optimization
```python
# Half precision for faster inference (on newer GPUs)
model = model.half()
image_tensor = image_tensor.half()

# Quantization
from torch.quantization import quantize_dynamic
model_int8 = quantize_dynamic(
    model, {nn.Linear},
    dtype=torch.qint8
)

# Export to ONNX for production
import torch.onnx
torch.onnx.export(
    model,
    image_tensor,
    "resnet50.onnx",
    input_names=['input'],
    output_names=['output']
)
```

## Error Handling

```python
def safe_predict(image_path, classifier):
    """Safely predict with error handling"""
    try:
        # Check file exists
        if not os.path.exists(image_path):
            return {'error': f"File not found: {image_path}"}
        
        # Check file is readable
        try:
            image = Image.open(image_path)
            image.verify()
        except Exception as e:
            return {'error': f"Invalid image file: {e}"}
        
        # Predict
        result = classifier.predict(image_path)
        return result
    
    except torch.cuda.OutOfMemoryError:
        return {'error': "GPU out of memory"}
    except Exception as e:
        return {'error': f"Prediction failed: {e}"}
```

---

## Summary

Complete ResNet inference pipeline:
1. Load pre-trained model
2. Create preprocessing pipeline
3. Load and preprocess image
4. Run inference
5. Convert logits to probabilities
6. Extract predictions
7. Visualize results

Key code patterns to remember:
- Always use `.eval()` mode for inference
- Always preprocess consistently
- Always use `torch.no_grad()` for inference
- Always normalize with ImageNet statistics
- Always add batch dimension

---

## Next Steps

- **Compare with ViT** → [Vision Transformer Implementation](../vit/vit-implementation.md)
- **Run Notebook** → [Notebooks Guide](../notebooks-guide.md)
- **Architecture Details** → [ResNet Overview](resnet-overview.md)
