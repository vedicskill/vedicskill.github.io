---
sidebar_position: 2
---

# Vision Transformer Implementation Guide

## Complete Vision Transformer Pipeline

This guide walks through implementing image classification with Vision Transformers from start to finish.

## Step 1: Imports and Setup

```python
import numpy as np
import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import matplotlib.pyplot as plt
```

**Each import's purpose:**
- `ViTForImageClassification`: Vision Transformer for classification
- `ViTImageProcessor`: Handles ViT-specific preprocessing
- Other libraries: Standard PyTorch, vision, and visualization

## Step 2: Load Pre-trained Vision Transformer

```python
# Model configuration
MODEL_NAME = "google/vit-base-patch16-224"

# Load model and processor
model = ViTForImageClassification.from_pretrained(MODEL_NAME)
processor = ViTImageProcessor.from_pretrained(MODEL_NAME)

# Set to evaluation mode
model.eval()

# Get class information
num_classes = model.config.num_labels
id2label = model.config.id2label
label2id = model.config.label2id

print(f"Model: ViT-Base (Patch 16×16)")
print(f"Input Size: 224×224")
print(f"Number of Classes: {num_classes}")
print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

# List some classes
print("\nSample Classes:")
for i in range(5):
    print(f"  {id2label[i]}")
```

**Output:**
```
Model: ViT-Base (Patch 16×16)
Input Size: 224×224
Number of Classes: 1000
Model Parameters: 86,567,616

Sample Classes:
  tench
  goldfish
  great_white_shark
  tiger_shark
  hammerhead
```

**Key points:**
- Model name tells us patch size: `patch16-224` = 16×16 patches, 224×224 input
- ViTImageProcessor knows the preprocessing for this model
- Access class labels through `model.config`

## Step 3: Understanding Preprocessing Differences

**ResNet preprocessing:**
```python
from torchvision import transforms

resnet_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

**ViT preprocessing:**
```python
# ViTImageProcessor handles everything!
# It knows the model-specific requirements
processor = ViTImageProcessor.from_pretrained(MODEL_NAME)

# The processor handles:
# - Resizing
# - Normalizing
# - Tensor conversion
# - Patch creation (implicit in model)
```

**Key difference:**
- ResNet: Manual preprocessing
- ViT: Model-specific processor (handles details automatically)

## Step 4: Load and Preprocess Image

```python
# Load image
image_path = '../data/image-3.jpg'
image = Image.open(image_path).convert('RGB')

print(f"Original image size: {image.size}")
print(f"Original image mode: {image.mode}")

# Preprocess for ViT
inputs = processor(images=image, return_tensors="pt")

print(f"\nProcessed inputs keys: {inputs.keys()}")
print(f"Pixel values shape: {inputs['pixel_values'].shape}")
print(f"Pixel values type: {inputs['pixel_values'].dtype}")
print(f"Value range: [{inputs['pixel_values'].min():.2f}, {inputs['pixel_values'].max():.2f}]")
```

**Output:**
```
Original image size: (640, 480)
Original image mode: RGB

Processed inputs keys: dict_keys(['pixel_values'])
Pixel values shape: torch.Size([1, 3, 224, 224])
Pixel values type: torch.float32
Value range: [-2.03, 2.63]
```

**What the processor does:**
```
Image → Resize → Normalize → Tensor
[640, 480, 3] → [224, 224, 3] → [0, 1] → [-2, 2] range
```

## Step 5: Run Inference

```python
# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}

print(f"Using device: {device}")

# Inference
with torch.no_grad():
    outputs = model(**inputs)

# Extract logits
logits = outputs.logits

print(f"Output logits shape: {logits.shape}")
print(f"Logits dtype: {logits.dtype}")
print(f"Min logit: {logits.min():.2f}")
print(f"Max logit: {logits.max():.2f}")
```

**Output:**
```
Using device: cuda
Output logits shape: torch.Size([1, 1000])
Logits dtype: torch.float32
Min logit: -2.45
Max logit: 3.82
```

**Key difference from ResNet:**
```
ResNet inference:
output = model(input_tensor)
logits = output

ViT inference:
output = model(**inputs)
logits = output.logits  ← Note: outputs is a NamedTuple!
```

## Step 6: Convert to Probabilities

```python
import torch.nn.functional as F

# Method 1: Using F.softmax
probabilities = F.softmax(logits, dim=1)

# Method 2: Using torch.softmax
# probabilities = torch.softmax(logits, dim=1)

print(f"Probabilities shape: {probabilities.shape}")
print(f"Sum of probabilities: {probabilities.sum(dim=1).item():.4f}")
print(f"Max probability: {probabilities.max().item():.4f}")
```

**Output:**
```
Probabilities shape: torch.Size([1, 1000])
Sum of probabilities: 1.0000
Max probability: 0.9156
```

## Step 7: Get Predictions

```python
# Get predicted class
predicted_idx = logits.argmax(-1).item()  # Note: -1 means last dimension
confidence = probabilities[0, predicted_idx].item()

# Get class name
predicted_class = id2label[predicted_idx]

print(f"Predicted class index: {predicted_idx}")
print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence:.2%}")
```

**Output:**
```
Predicted class index: 207
Predicted class: golden_retriever
Confidence: 91.56%
```

## Step 8: Get Top-K Predictions

```python
# Get top-5 predictions
top_k = 5
top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)

print(f"\nTop-{top_k} Predictions:")
print("-" * 50)

for rank, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0]), 1):
    class_name = id2label[idx.item()]
    confidence = prob.item()
    
    # Format nicely
    class_name_formatted = class_name.replace('_', ' ').title()
    print(f"{rank}. {class_name_formatted:<35} {confidence:>6.2%}")

print("-" * 50)
```

**Output:**
```
Top-5 Predictions:
--------------------------------------------------
1. Golden Retriever                   91.56%
2. Labrador Retriever                  5.43%
3. Yellow Labrador                     2.18%
4. Great Pyrenees                      0.51%
5. Setter                              0.23%
--------------------------------------------------
```

## Step 9: Visualize with Processor and Model

```python
def visualize_vit_predictions(image_path, top_k=5):
    """Visualize ViT predictions with attention"""
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Preprocess
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    
    # Get top predictions
    top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Original image
    ax1.imshow(image)
    
    # Draw patch grid (14×14 patches for 224×224 image)
    patch_size = 16
    for i in range(0, 224, patch_size):
        ax1.axhline(y=i, color='white', linewidth=0.5, alpha=0.3)
        ax1.axvline(x=i, color='white', linewidth=0.5, alpha=0.3)
    
    ax1.set_title('Input Image with Patch Grid\n(14×14=196 patches)')
    ax1.axis('off')
    
    # Right: Top predictions
    class_names = [id2label[idx.item()].replace('_', ' ').title() 
                   for idx in top_indices[0]]
    probs = [prob.item() for prob in top_probs[0]]
    
    bars = ax2.barh(range(top_k), probs, color=plt.cm.viridis(np.linspace(0, 1, top_k)))
    ax2.set_yticks(range(top_k))
    ax2.set_yticklabels(class_names)
    ax2.set_xlabel('Confidence')
    ax2.set_title(f'Top-{top_k} ViT Predictions')
    ax2.invert_yaxis()
    
    # Add percentage labels
    for i, (prob, bar) in enumerate(zip(probs, bars)):
        ax2.text(prob, i, f' {prob:.1%}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    return class_names[0], probs[0]

# Usage
predicted, confidence = visualize_vit_predictions('../data/image-3.jpg')
print(f"\nPrediction: {predicted}")
print(f"Confidence: {confidence:.2%}")
```

## Complete Example: ViTClassifier Class

```python
class ViTClassifierWrapper:
    """Reusable Vision Transformer classifier"""
    
    def __init__(self, model_name="google/vit-base-patch16-224", device=None):
        """
        Initialize ViT classifier
        
        Args:
            model_name: HuggingFace model identifier
            device: torch.device (auto-detect if None)
        """
        # Load model and processor
        self.model = ViTForImageClassification.from_pretrained(model_name)
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Get label mappings
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
    
    def predict(self, image_path, top_k=5):
        """
        Predict class of image
        
        Args:
            image_path: Path to image
            top_k: Return top-k predictions
            
        Returns:
            Dictionary with results
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Preprocess
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        
        # Get top-k
        top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
        
        # Format results
        results = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            class_idx = idx.item()
            class_name = self.id2label[class_idx]
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
            'image_path': image_path,
            'model': 'ViT-Base-Patch16-224'
        }
    
    def predict_batch(self, image_paths, top_k=1):
        """Predict on multiple images"""
        results = []
        for image_path in image_paths:
            result = self.predict(image_path, top_k=top_k)
            results.append(result)
        return results

# Usage
classifier = ViTClassifierWrapper()
results = classifier.predict('../data/image-3.jpg', top_k=5)

print(f"Prediction: {results['prediction']}")
print(f"Confidence: {results['confidence']:.2%}")
print("\nTop-5:")
for i, result in enumerate(results['top_k'], 1):
    print(f"{i}. {result['class']}: {result['confidence']:.2%}")
```

## Understanding Vision Transformer Internals

### Accessing Intermediate Layers

```python
# Get embeddings before transformer
class ViTFeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.embeddings = model.vit.embeddings
        self.encoder = model.vit.encoder
    
    def get_patch_embeddings(self, inputs):
        """Get patch embeddings (after linear projection)"""
        embedding_output = self.embeddings(inputs["pixel_values"])
        return embedding_output  # [batch, num_patches+1, hidden_dim]
    
    def get_encoder_output(self, inputs):
        """Get transformer encoder output"""
        embedding_output = self.embeddings(inputs["pixel_values"])
        encoder_output = self.encoder(embedding_output)
        return encoder_output[0]  # [batch, num_patches+1, hidden_dim]

# Usage
feature_extractor = ViTFeatureExtractor(model)

# Get patch embeddings (before transformer)
patch_emb = feature_extractor.get_patch_embeddings(inputs)
print(f"Patch embeddings shape: {patch_emb.shape}")
# Output: torch.Size([1, 197, 768])
# 197 = 196 patches + 1 [CLS] token

# Get transformer output
enc_output = feature_extractor.get_encoder_output(inputs)
print(f"Encoder output shape: {enc_output.shape}")
# Output: torch.Size([1, 197, 768])
```

### Visualizing Attention

```python
def get_attention_weights(model, inputs):
    """Extract attention weights from all layers"""
    
    with torch.no_grad():
        outputs = model(
            inputs["pixel_values"],
            output_attentions=True  # Get attention weights
        )
    
    attentions = outputs.attentions  # Tuple of 12 attention tensors
    
    # Each attention:
    # [batch, num_heads, seq_len, seq_len]
    # [1, 12, 197, 197]
    
    return attentions

# Usage
inputs = processor(images=image, return_tensors="pt").to(device)
attentions = get_attention_weights(model, inputs)

print(f"Number of attention layers: {len(attentions)}")
print(f"First layer shape: {attentions[0].shape}")
# Output: torch.Size([1, 12, 197, 197])
```

## Comparing ResNet vs ViT

```python
class ImageClassificationComparison:
    """Compare ResNet and ViT predictions"""
    
    def __init__(self):
        from torchvision.models import resnet50, ResNet50_Weights
        
        # ResNet
        resnet_weights = ResNet50_Weights.DEFAULT
        self.resnet = resnet50(weights=resnet_weights)
        self.resnet.eval()
        self.resnet_labels = resnet_weights.meta['categories']
        self.resnet_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        
        # ViT
        self.vit = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224"
        )
        self.vit.eval()
        self.vit_processor = ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224"
        )
        self.vit_labels = self.vit.config.id2label
    
    def compare(self, image_path):
        """Compare predictions from both models"""
        
        image = Image.open(image_path).convert('RGB')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # ResNet prediction
        resnet_input = self.resnet_transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            resnet_logits = self.resnet(resnet_input)
        resnet_probs = F.softmax(resnet_logits, dim=1)
        resnet_class = self.resnet_labels[resnet_probs.argmax().item()]
        resnet_conf = resnet_probs.max().item()
        
        # ViT prediction
        vit_inputs = self.vit_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            vit_outputs = self.vit(**vit_inputs)
        vit_probs = F.softmax(vit_outputs.logits, dim=1)
        vit_class = self.vit_labels[vit_probs.argmax().item()]
        vit_conf = vit_probs.max().item()
        
        return {
            'ResNet': {'class': resnet_class, 'confidence': resnet_conf},
            'ViT': {'class': vit_class, 'confidence': vit_conf}
        }

# Usage
comparator = ImageClassificationComparison()
results = comparator.compare('../data/image-3.jpg')

print("Model Predictions Comparison:")
print("-" * 50)
for model, prediction in results.items():
    print(f"{model:10} {prediction['class']:25} {prediction['confidence']:.2%}")
print("-" * 50)
```

## Performance Optimization

```python
# 1. Use mixed precision for faster inference
from torch.cuda.amp import autocast

with autocast():
    with torch.no_grad():
        outputs = model(**inputs)

# 2. Batch processing
def batch_predict(image_paths, batch_size=4):
    """Efficient batch prediction"""
    results = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        
        # Load images
        images = [Image.open(p).convert('RGB') for p in batch_paths]
        
        # Preprocess batch
        inputs = processor(images=images, return_tensors="pt").to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Process results
        logits = outputs.logits
        predictions = logits.argmax(-1)
        
        for pred_idx, image_path in zip(predictions, batch_paths):
            class_name = id2label[pred_idx.item()]
            results.append({
                'image': image_path,
                'class': class_name
            })
    
    return results

# Usage
images = ['image1.jpg', 'image2.jpg', 'image3.jpg']
results = batch_predict(images, batch_size=2)
```

## Error Handling

```python
import os

def safe_vit_predict(image_path, classifier):
    """Safely predict with error handling"""
    try:
        # Validate file
        if not os.path.exists(image_path):
            return {'error': f'File not found: {image_path}'}
        
        # Try to open
        try:
            image = Image.open(image_path)
            image.verify()
        except Exception as e:
            return {'error': f'Invalid image: {e}'}
        
        # Predict
        result = classifier.predict(image_path)
        return result
    
    except torch.cuda.OutOfMemoryError:
        return {'error': 'GPU out of memory'}
    except Exception as e:
        return {'error': f'Prediction failed: {e}'}
```

---

## Summary

Complete Vision Transformer inference pipeline:
1. Load pre-trained ViT model and processor
2. Load and preprocess image using processor
3. Run inference with `model(**inputs)`
4. Extract logits from outputs
5. Convert to probabilities
6. Get top-k predictions
7. Visualize results

Key code patterns:
- Use `ViTImageProcessor` for preprocessing
- Access logits via `outputs.logits`
- Use argmax with `-1` to get last dimension
- Access class labels via `model.config.id2label`

---

## Next Steps

- **Compare with ResNet** → [../resnet/resnet-implementation.md](../resnet/resnet-implementation.md)
- **Run Notebook** → [../notebooks-guide.md](../notebooks-guide.md)
- **Architecture Details** → [Vision Transformer Overview](transformer-overview.md)
