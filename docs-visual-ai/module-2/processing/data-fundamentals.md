---
sidebar_position: 1
---

# Data Fundamentals for Image

## Understanding Images as Data

### What is an Image?

An image is a 2D array of pixel values. Each pixel represents a small point in the image with color information.

```
Image = 3D Tensor

Dimensions: (Height, Width, Channels)
           = (H, W, C)

Example: Standard image
        (224, 224, 3)
        - Height: 224 pixels
        - Width: 224 pixels  
        - Channels: 3 (Red, Green, Blue)
```

### Pixel Values

Each pixel is represented by numerical values:

**Grayscale Image (1 channel):**
```
Pixel value: 0-255
0 = Black
127 = Mid-gray
255 = White
```

**RGB Image (3 channels):**
```
Each pixel has 3 values: [R, G, B]
[255, 0, 0]     = Pure Red
[0, 255, 0]     = Pure Green
[0, 0, 255]     = Pure Blue
[255, 255, 255] = White
[0, 0, 0]       = Black
```

### Image Formats

Common image file formats:

- **JPEG** - Lossy compression, smaller file size, most common
- **PNG** - Lossless compression, supports transparency
- **TIFF** - High quality, large file size, scientific imaging
- **BMP** - Uncompressed, large file size
- **WebP** - Modern format, good compression

## Image Loading and Basic Operations

### Loading an Image in Python

```python
import numpy as np
from PIL import Image
import cv2

# Using PIL
image = Image.open('dog.jpg')
print(image.size)  # (width, height)
print(image.mode)  # 'RGB' or 'RGBA'

# Using OpenCV
image_cv = cv2.imread('dog.jpg')
print(image_cv.shape)  # (height, width, channels)

# Using PyTorch
from torchvision import transforms
image_tensor = transforms.ToTensor()(Image.open('dog.jpg'))
print(image_tensor.shape)  # (channels, height, width)
```

### Image Shapes in Different Libraries

```python
# PIL.Image
image.size  # (width, height)

# NumPy (OpenCV)
image.shape  # (height, width, channels)

# PyTorch (TorchVision)
tensor.shape  # (channels, height, width) or (batch, channels, height, width)
```

### Basic Image Manipulations

```python
from PIL import Image
import numpy as np

# Load image
image = Image.open('dog.jpg')

# Convert to numpy array
image_array = np.asarray(image)
print(image_array.shape)  # (H, W, C)

# Get image statistics
print(f"Min pixel value: {image_array.min()}")
print(f"Max pixel value: {image_array.max()}")
print(f"Mean: {image_array.mean()}")
print(f"Std: {image_array.std()}")

# Crop image
cropped = image.crop((left, top, right, bottom))

# Rotate image
rotated = image.rotate(45)  # degrees

# Flip image
flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
```

## Standard Image Sizes

### Common Sizes for Classification

| Model | Size | Aspect | Use Case |
|-------|------|--------|----------|
| ResNet | 224×224 | Square | Standard, fast |
| ViT-Base | 224×224 | Square | Standard transformer |
| Inception | 299×299 | Square | Better detail |
| EfficientNet | 224-600 | Variable | Flexible |
| CLIP ViT | 224×224 | Square | Vision-language |

### Why 224×224?

```
224 = 7 × 2^5

Historical reasons:
1. AlexNet used 256×256, then cropped to 224×224
2. 224 is convenient power-of-2 related
3. Provides good balance between detail and computation
4. Standard for ImageNet training

But any size works theoretically!
For example: 384×384, 512×512, 768×768
Larger sizes → more detail but slower inference
```

## Preprocessing Pipeline

### Standard Preprocessing Steps

```python
from torchvision import transforms

# Create preprocessing pipeline
transform = transforms.Compose([
    # Step 1: Resize to standard size
    transforms.Resize((224, 224)),
    
    # Step 2: Convert PIL Image to Tensor
    # Converts pixel values from [0, 255] to [0, 1]
    transforms.ToTensor(),
    
    # Step 3: Normalize using ImageNet statistics
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # Pre-computed ImageNet means
        std=[0.229, 0.224, 0.225]    # Pre-computed ImageNet stds
    )
])

# Apply to image
from PIL import Image
image = Image.open('dog.jpg')
processed = transform(image)
print(processed.shape)  # torch.Size([3, 224, 224])
```

### Understanding Normalization

**Why normalize?**
1. **Standardizes input** - Helps model training stability
2. **Faster convergence** - Gradients are more balanced
3. **Transfer learning** - Must use same normalization as pre-training

**ImageNet Normalization (Why these specific values?):**

```
Mean = [0.485, 0.456, 0.406]
Std  = [0.229, 0.224, 0.225]

These are computed from the ImageNet training dataset:
- All 1.2 million training images
- Per-channel statistics

Why use these?
- Pre-trained models expect this normalization
- Ensures compatibility with pre-trained weights
```

**How normalization works:**

```
Before normalization:
pixel_value ∈ [0, 255]

Step 1: ToTensor()
pixel_value ∈ [0, 1]

Step 2: Normalize (per channel)
normalized = (pixel - mean) / std

For R channel (mean=0.485, std=0.229):
0.5 (red intensity) → (0.5 - 0.485) / 0.229 = 0.065

Result:
normalized_value ∈ [-2, 2] (approximately)
```

## Data for Training vs Inference

### Training Data Requirements

```
Image Classification Training:
- Millions of labeled images (1M+)
- Diverse categories (1000+ classes for ImageNet)
- Balanced distribution (similar number per class)
- High quality annotations

Example: ImageNet
- 1.2 million training images
- 50,000 validation images
- 1000 categories
- Carefully curated and verified labels
```

### Inference Data (What We Use)

```
For inference (prediction on new images):
- Any image size (will be resized)
- Any format (JPEG, PNG, etc.)
- No labels needed (we predict them)
- Can be a single image or batch

In this course:
- We use pre-trained models
- Already trained on millions of images
- We just predict on new images
```

## ImageNet: The Foundation Dataset

### What is ImageNet?

ImageNet is the dataset that transformed computer vision. Understanding it helps understand standard practices.

```
ImageNet Dataset:
- 1.2 million training images
- 50,000 validation images
- 1,000 categories
- Hand-labeled and verified
- Hierarchically organized (synsets)

Example categories:
- 119 dog breeds (poodle, husky, etc.)
- 27 vehicle types (car, truck, bus, etc.)
- 13 furniture categories (chair, sofa, etc.)
- And 961 more...
```

### ImageNet Preprocessing Standard

All models trained on ImageNet use:

```python
# STANDARD for ImageNet models:
from torchvision import transforms

transform = transforms.Compose([
    # Resize shorter edge to 256, maintain aspect ratio
    transforms.Resize(256),
    
    # Crop center 224×224
    transforms.CenterCrop(224),
    
    # Convert to tensor
    transforms.ToTensor(),
    
    # Normalize with ImageNet statistics
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# For inference/testing - deterministic
# For training - typically add augmentation
```

## Data Augmentation

Data augmentation creates variations of images during training to improve model robustness.

### Common Augmentation Techniques

```python
from torchvision import transforms

# Training augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),      # Random crop with zoom
    transforms.RandomHorizontalFlip(),      # 50% chance flip
    transforms.RandomRotation(10),          # ±10 degrees rotation
    transforms.ColorJitter(                 # Vary color
        brightness=0.1,
        contrast=0.1,
        saturation=0.1,
        hue=0.05
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Test augmentation (no random transformations)
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

### Why Augmentation Helps

```
Original training set: 1000 images per class

With augmentation: Effectively unlimited variations
- Flipped versions
- Rotated versions
- Zoomed-in crops
- Color variations
- Combinations of above

Benefits:
1. Model sees more diversity
2. Better generalization
3. More robust to variations
4. Higher final accuracy
```

## Batching Images

Models process images in batches for efficiency.

```python
# Single image
image.shape  # [3, 224, 224]

# Batch of 32 images
batch.shape  # [32, 3, 224, 224]
                # ^ batch dimension

# Processing benefits:
# - Parallelization (GPU handles all at once)
# - Better memory utilization
# - 32-64 images processed in similar time as 1 image
```

### Creating Batches in Code

```python
import torch
from torchvision import transforms
from PIL import Image

# Load and preprocess single image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Single image
image = Image.open('dog.jpg')
image_tensor = transform(image)
print(image_tensor.shape)  # [3, 224, 224]

# Create batch of 1
batch = image_tensor.unsqueeze(0)
print(batch.shape)  # [1, 3, 224, 224]

# Create batch of 32 by stacking
images = [transform(Image.open(f'image_{i}.jpg')) for i in range(32)]
batch = torch.stack(images)
print(batch.shape)  # [32, 3, 224, 224]
```

## Dataset Organization

### Typical Project Structure

```
project/
├── data/
│   ├── images/
│   │   ├── image-1.jpg
│   │   ├── image-2.jpg
│   │   └── ...
│   ├── labels/
│   │   ├── labels.csv
│   │   └── labels.json
│   ├── splits/
│   │   ├── train_images.txt
│   │   ├── val_images.txt
│   │   └── test_images.txt
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_resnet_classification.ipynb
│   └── 03_transformer_classification.ipynb
```

### Metadata for Images

```python
# CSV format (labels.csv)
image_id,category
image-1.jpg,dog
image-2.jpg,cat
image-3.jpg,car
...

# JSON format (labels.json)
{
  "image-1.jpg": "dog",
  "image-2.jpg": "cat",
  "image-3.jpg": "car"
}

# Reading in code
import pandas as pd
df = pd.read_csv('labels.csv')
labels = dict(zip(df['image_id'], df['category']))
```

## Summary of Key Points

1. **Images are tensors** - Height × Width × Channels
2. **Standardize sizes** - Usually 224×224 for classification models
3. **Normalize values** - Use ImageNet normalization for pre-trained models
4. **Preprocess consistently** - Same transform for inference as training
5. **Batch for efficiency** - Process multiple images together
6. **Augmentation helps** - Improves training when data is limited

---

## Next Steps

- **Image Processing Techniques** → [Image Processing](image-processing.md)
- **See Data in Practice** → [Notebooks Guide](../notebooks-guide.md)
- **Explore Implementation** → [ResNet Implementation](../resnet/resnet-implementation.md)
