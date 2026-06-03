---
sidebar_position: 2
---


# Image Processing for Classification

## Image Processing Basics

Image processing transforms raw images into formats suitable for machine learning models. This section covers essential techniques used in the classification pipeline.

## Common Image Processing Operations

### 1. Resizing Images

Resizing standardizes image dimensions for model input.

```python
from PIL import Image
import cv2

# Using PIL
image = Image.open('image.jpg')
print(image.size)  # (width, height)

resized = image.resize((224, 224))  # (width, height)
print(resized.size)  # (224, 224)

# Using OpenCV
image_cv = cv2.imread('image.jpg')
print(image_cv.shape)  # (height, width, channels)

resized_cv = cv2.resize(image_cv, (224, 224))  # (width, height)
print(resized_cv.shape)  # (224, 224, 3)

# Using PyTorch transforms
from torchvision import transforms
transform = transforms.Resize((224, 224))
```

### Resizing Methods

Different approaches for resizing:

```python
from PIL import Image

image = Image.open('image.jpg')  # Original: 640×480

# Method 1: Direct resize (may distort aspect ratio)
direct = image.resize((224, 224))
# Result: 224×224 (may look stretched)

# Method 2: Aspect-ratio preserving with letterboxing
# Resize smaller edge to 224, keep aspect ratio
image.thumbnail((224, 224), Image.Resampling.LANCZOS)
# Add padding to make it 224×224
# Result: 224×224 with black borders if needed

# Method 3: Center crop after resize
# Resize to 256, then crop center 224×224
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224)
])
# Result: 224×224 (no distortion, no padding)
```

### 2. Color Space Conversion

Images can be in different color spaces.

```python
import cv2
from PIL import Image

image = cv2.imread('image.jpg')

# OpenCV loads as BGR by default
# Convert to RGB
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(gray.shape)  # (height, width) - single channel

# Convert to HSV (Hue, Saturation, Value)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Convert to LAB color space
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
```

### Why Color Space Matters

```
RGB: Red, Green, Blue
- Human perception
- Most common for images
- Used in most deep learning

BGR: Blue, Green, Red
- OpenCV default (historical reason)
- Need to convert to RGB for models

Grayscale: Single intensity channel
- Medical imaging
- Some specialized tasks
- Typically not used for classification

HSV: Hue, Saturation, Value
- More intuitive for color manipulation
- Used in image processing
- Sometimes used for augmentation
```

### 3. Normalization

Normalize pixel values to standard ranges.

```python
import numpy as np
from PIL import Image

image = Image.open('image.jpg')
image_array = np.asarray(image, dtype=np.float32)

# Original range: [0, 255]

# Normalize to [0, 1]
normalized = image_array / 255.0

# Standardize: subtract mean, divide by std
mean = np.array([0.485, 0.456, 0.406])  # ImageNet mean
std = np.array([0.229, 0.224, 0.225])   # ImageNet std
standardized = (normalized - mean) / std

# Result: approximately [-2, 2] range
```

### When to Normalize

```
Training phase:
- Always normalize!
- Stabilizes gradients
- Faster convergence

Inference phase:
- Always normalize!
- Must match training normalization
- Different values ≠ correct predictions

Standard for ImageNet models:
- ToTensor() converts [0, 255] to [0, 1]
- Then Normalize() applies mean/std
```

### 4. Contrast and Brightness Adjustment

Adjust image appearance for better learning.

```python
import cv2
import numpy as np

image = cv2.imread('image.jpg')

# Brightness adjustment
brightness = 30
bright_image = cv2.convertScaleAbs(image, alpha=1.0, beta=brightness)

# Contrast adjustment (using CLAHE - Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
contrast_adjusted = clahe.apply(gray)

# Simple contrast scaling
contrast = 1.5
contrast_image = cv2.convertScaleAbs(image, alpha=contrast, beta=0)
```

### 5. Filtering and Smoothing

Reduce noise while preserving edges.

```python
import cv2

image = cv2.imread('image.jpg')

# Gaussian Blur - smooth noise
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Bilateral Filter - smooth while preserving edges
bilateral = cv2.bilateralFilter(image, 9, 75, 75)

# Median Filter - good for salt-and-pepper noise
median = cv2.medianBlur(image, 5)

# Morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
eroded = cv2.erode(image, kernel)
dilated = cv2.dilate(image, kernel)
```

### 6. Edge Detection

Detect image boundaries and features.

```python
import cv2

image = cv2.imread('image.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Sobel edge detection
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

# Canny edge detection (more sophisticated)
edges = cv2.Canny(gray, 100, 200)

# Laplacian
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
```

## Complete Preprocessing Pipeline

### Step-by-Step Example

```python
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

# Step 1: Load image
image = Image.open('dog.jpg')
print(f"Original size: {image.size}")  # (640, 480)
print(f"Mode: {image.mode}")  # RGB

# Step 2: Define preprocessing
preprocess = transforms.Compose([
    # Resize to 256×256 (keeping aspect ratio)
    transforms.Resize(256),
    
    # Crop center 224×224
    transforms.CenterCrop(224),
    
    # Convert to tensor ([0, 1] range)
    transforms.ToTensor(),
    
    # Normalize with ImageNet statistics
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Step 3: Apply preprocessing
tensor = preprocess(image)
print(f"Final shape: {tensor.shape}")  # torch.Size([3, 224, 224])
print(f"Value range: [{tensor.min():.2f}, {tensor.max():.2f}]")

# Step 4: Add batch dimension for model
batch = tensor.unsqueeze(0)
print(f"Batch shape: {batch.shape}")  # torch.Size([1, 3, 224, 224])
```

## Handling Different Image Formats

### JPEG Images

```python
from PIL import Image
import torch
from torchvision import transforms

# Load JPEG
image = Image.open('photo.jpg')

# JPEG is lossy, but standard for photos
# Quality is typically fine for classification

# Process
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
tensor = transform(image)
```

### PNG Images

```python
from PIL import Image

# Load PNG
image = Image.open('image.png')

# PNG supports transparency (RGBA)
if image.mode == 'RGBA':
    # Convert RGBA to RGB (remove alpha channel)
    rgb_image = image.convert('RGB')
else:
    rgb_image = image

# Process as normal
# ... same as JPEG processing
```

### Medical/Scientific Images

```python
import cv2
import numpy as np

# Load TIFF (common for medical)
image = cv2.imread('medical.tif', cv2.IMREAD_GRAYSCALE)

# Often 16-bit or higher
# Need to normalize to [0, 255] or [0, 1]
if image.dtype == np.uint16:
    image_normalized = (image / 65535.0 * 255).astype(np.uint8)
elif image.dtype == np.float32 or image.dtype == np.float64:
    image_normalized = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)
else:
    image_normalized = image
```

## Handling Edge Cases

### 1. Corrupted or Missing Images

```python
import os
from PIL import Image
from torchvision import transforms

def safe_load_image(image_path):
    """Safely load image with error handling"""
    try:
        image = Image.open(image_path)
        
        # Verify it's a valid image
        image.verify()
        
        # Reopen after verify (verify closes it)
        image = Image.open(image_path)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None

# Usage
image = safe_load_image('image.jpg')
if image is not None:
    # Process image
    pass
else:
    # Handle missing/corrupted image
    pass
```

### 2. Different Sized Images

```python
from PIL import Image
from torchvision import transforms
import torch

def load_and_process_batch(image_paths):
    """Load images of different sizes and create batch"""
    images = []
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize all to same size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    for path in image_paths:
        try:
            image = Image.open(path).convert('RGB')
            tensor = transform(image)
            images.append(tensor)
        except Exception as e:
            print(f"Skipping {path}: {e}")
            continue
    
    if images:
        batch = torch.stack(images)
        return batch
    else:
        return None

# Usage
batch = load_and_process_batch(['image1.jpg', 'image2.jpg', 'image3.jpg'])
```

### 3. Very Large Images

```python
from PIL import Image
from torchvision import transforms

def process_large_image(image_path, target_size=224):
    """Handle large images efficiently"""
    image = Image.open(image_path)
    
    # Get original size
    width, height = image.size
    
    # Resize if larger than needed
    if width > target_size * 2 or height > target_size * 2:
        # Reduce to 2x target size first (faster than direct resize)
        image.thumbnail((target_size * 2, target_size * 2), Image.Resampling.LANCZOS)
    
    # Final resize and crop
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image)
```

## Batch Processing

### Efficient Batch Loading

```python
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

class ImageDataset(Dataset):
    """Custom dataset for loading images"""
    
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = os.listdir(image_dir)
        self.transform = transform
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image

# Setup
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

dataset = ImageDataset('data/images', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Use in inference
for batch in dataloader:
    print(batch.shape)  # [32, 3, 224, 224]
    # Process batch
```

## Summary

Key image processing concepts:
1. **Resizing** - Standardize dimensions
2. **Color spaces** - Convert between RGB, BGR, Grayscale
3. **Normalization** - Scale values appropriately
4. **Filtering** - Reduce noise
5. **Edge detection** - Identify features
6. **Batch processing** - Efficient inference

For classification, the essential steps are:
1. Load image
2. Resize to 224×224 (or appropriate size)
3. Convert to tensor
4. Normalize with ImageNet statistics
5. Add batch dimension
6. Send to model

---

## Next Steps

- **Data Fundamentals** → [Data Fundamentals](data-fundamentals.md)
- **See in Practice** → [Notebooks Guide](../notebooks-guide.md)
