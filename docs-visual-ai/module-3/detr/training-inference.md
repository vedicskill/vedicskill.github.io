---
sidebar_position: 5
title: DETR Training and Inference
description: How to train DETR models and perform inference on new images
---

# DETR Training and Inference

## Training DETR

### Training Pipeline

```
Dataset (COCO, etc.)
        │
        ▼
Data Loading & Augmentation
        │
        ▼
Batch Preparation
        │
        ├─ Images: (batch_size, 3, H, W)
        ├─ Targets: [class, bbox] × objects
        │
        ▼
Forward Pass
        │
        ├─ Output: predictions
        ├─ Target: ground truth
        │
        ▼
Hungarian Matching
        │
        └─ Match predictions to ground truth
        │
        ▼
Loss Computation
        │
        ├─ Classification Loss
        ├─ L1 Loss (bounding box)
        ├─ GIoU Loss (bounding box)
        │
        ▼
Backpropagation
        │
        ├─ Compute gradients
        │
        ▼
Optimizer Update
        │
        ├─ Update weights
        ├─ Especially update object queries!
        │
        ▼
Repeat for all batches
```

### Key Training Concepts

#### 1. Hungarian Matching

**Problem:** During training, we need to know:
- Which predicted object corresponds to which ground truth?
- How to compute loss fairly?

**Solution:** Use Hungarian Algorithm for optimal matching.

```python
# Pseudocode
def hungarian_matching(predictions, targets):
    """
    predictions: (batch_size, 100, num_classes + 4)
    targets: list of ground truth objects
    """
    
    matches = []
    
    for batch_idx in range(batch_size):
        pred = predictions[batch_idx]      # (100, num_classes + 4)
        target = targets[batch_idx]        # variable length
        
        # Compute cost matrix
        # cost[i][j] = cost of matching pred[i] to target[j]
        cost_matrix = compute_cost(pred, target)
        
        # Find optimal matching (Hungarian algorithm)
        matching = hungarian_algorithm(cost_matrix)
        
        matches.append(matching)
    
    return matches

# Cost computation
def compute_cost(pred, target):
    """
    Cost includes:
    1. Classification mismatch (CE loss)
    2. Bounding box mismatch (L1 + GIoU)
    """
    
    cost = 0
    
    for pred_idx in range(100):
        for target_idx in range(len(target)):
            # Class cost
            class_cost = cross_entropy(pred[pred_idx], target[target_idx])
            
            # Bbox cost (L1 + GIoU)
            bbox_cost = l1_loss(...) + giou_loss(...)
            
            cost[pred_idx][target_idx] = class_cost + bbox_cost
    
    return cost
```

#### 2. Loss Function

```python
def detr_loss(predictions, targets, matches):
    """
    predictions: (batch_size, 100, num_classes + 4)
    targets: list of ground truth
    matches: Hungarian matching output
    """
    
    total_loss = 0
    
    for batch_idx in range(batch_size):
        pred = predictions[batch_idx]
        target = targets[batch_idx]
        matching = matches[batch_idx]
        
        # For each matched pair
        for pred_idx, target_idx in matching:
            pred_class = pred[pred_idx, :num_classes]
            pred_bbox = pred[pred_idx, num_classes:]
            
            target_class = target[target_idx, 'class']
            target_bbox = target[target_idx, 'bbox']
            
            # Classification loss
            class_loss = cross_entropy(pred_class, target_class)
            
            # Bounding box loss
            l1_loss_val = l1_loss(pred_bbox, target_bbox)
            giou_loss_val = giou_loss(pred_bbox, target_bbox)
            bbox_loss = l1_loss_val + giou_loss_val
            
            # Weighted combination
            loss = class_loss + 5 * l1_loss_val + 2 * giou_loss_val
            
            total_loss += loss
        
        # For unmatched predictions (background)
        for pred_idx in unmatched_predictions:
            background_loss = cross_entropy(pred[pred_idx], background_class)
            total_loss += background_loss
    
    return total_loss / batch_size
```

#### 3. Loss Weights

```python
loss = lambda_ce * ce_loss + lambda_l1 * l1_loss + lambda_giou * giou_loss

Typical weights:
  lambda_ce = 1.0     (classification)
  lambda_l1 = 5.0     (localization - higher weight)
  lambda_giou = 2.0   (localization - higher weight)

Why higher weights for localization?
  - Need accurate bounding boxes
  - Help object queries learn spatial positions
```

### Training Challenges

#### Challenge 1: Slow Convergence

**Problem:**
```
YOLO: Converges in ~50 epochs
DETR: Requires ~300+ epochs!

Why?
- Object queries start randomly
- Need time to specialize
- Hungarian matching adds complexity
```

**Solution:**
```
1. Use warm-up learning rate
   LR schedule: 0 → initial LR → decay
   
2. Use pre-trained backbone
   ResNet-50 pre-trained on ImageNet
   
3. Auxiliary losses
   Add losses at intermediate decoder layers
   
4. Learning rate strategy
   Lower learning rate
   Longer training schedule
```

#### Challenge 2: Small Object Detection

**Problem:**
```
DETR initially weaker at small objects

Why?
- Encoder downsamples 32x (loses small detail)
- Limited feature resolution
```

**Solution:**
```
1. Use multi-scale features
   Process at multiple resolutions
   
2. Add FPN (Feature Pyramid Network)
   Combine multi-scale features
   
3. Increased training time
   More examples of small objects
```

### Training Code Example

```python
import torch
import torch.nn as nn
from torch.optim import AdamW
from detr import DETR

# Load model
model = DETR(num_classes=80)

# Move to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

# Learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Training loop
num_epochs = 300
batch_size = 16

for epoch in range(num_epochs):
    total_loss = 0
    
    for batch_images, batch_targets in train_loader:
        batch_images = batch_images.to(device)
        
        # Forward pass
        outputs = model(batch_images)
        
        # Hungarian matching
        matches = hungarian_matching(outputs, batch_targets)
        
        # Compute loss
        loss = compute_loss(outputs, batch_targets, matches)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()
        
        total_loss += loss.item()
    
    # Update learning rate
    lr_scheduler.step()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
    
    # Validation
    if (epoch + 1) % 10 == 0:
        validate(model, val_loader, device)
```

---

## Inference on New Images

### Inference Pipeline

```
New Image
        │
        ▼
Resize to standard size (e.g., 800×1200)
        │
        ▼
Normalize (subtract mean, divide by std)
        │
        ▼
Add batch dimension: (1, 3, 800, 1200)
        │
        ▼
Forward pass (no gradient computation)
        │
        ▼
Get predictions: (1, 100, 81) classes + (1, 100, 4) bboxes
        │
        ▼
Post-processing:
  ├─ Apply softmax to classes
  ├─ Filter by confidence threshold
  ├─ Optional NMS
        │
        ▼
Final Detections
```

### Inference Code Example

```python
import torch
from PIL import Image
from torchvision import transforms
import cv2

# Load trained model
model = DETR(num_classes=80)
model.load_state_dict(torch.load('detr_pretrained.pth'))
model.eval()  # Set to evaluation mode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((800, 1200)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load and preprocess image
image_path = 'image.jpg'
image = Image.open(image_path)
image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
image_tensor = image_tensor.to(device)

# Forward pass (no gradients)
with torch.no_grad():
    outputs = model(image_tensor)

# Extract predictions
logits = outputs['pred_logits']      # (1, 100, 81)
bboxes = outputs['pred_boxes']      # (1, 100, 4)

# Apply softmax to get probabilities
probabilities = torch.softmax(logits, dim=-1)

# Get confidence scores (max probability per query)
confidences, class_ids = torch.max(probabilities[0], dim=-1)

# Filter by confidence threshold
threshold = 0.5
mask = confidences > threshold

filtered_classes = class_ids[mask]
filtered_confidences = confidences[mask]
filtered_bboxes = bboxes[0][mask]

# Denormalize bounding boxes
def denormalize_bbox(bbox_norm, image_width, image_height):
    """
    Convert from normalized [cx, cy, w, h] to pixel [x1, y1, x2, y2]
    """
    cx, cy, w, h = bbox_norm
    
    x1 = (cx - w/2) * image_width
    y1 = (cy - h/2) * image_height
    x2 = (cx + w/2) * image_width
    y2 = (cy + h/2) * image_height
    
    return [x1, y1, x2, y2]

# Get image dimensions
image_width, image_height = image.size

# Denormalize all bboxes
detections = []
for i, (class_id, confidence, bbox) in enumerate(zip(
    filtered_classes, filtered_confidences, filtered_bboxes
)):
    bbox_pixel = denormalize_bbox(bbox.cpu().numpy(), image_width, image_height)
    
    detection = {
        'class_id': class_id.item(),
        'class_name': class_names[class_id.item()],
        'confidence': confidence.item(),
        'bbox': bbox_pixel
    }
    detections.append(detection)

# Optional: Apply NMS
def nms(detections, iou_threshold=0.5):
    """Apply Non-Maximum Suppression"""
    # ... NMS implementation ...
    return detections_after_nms

detections = nms(detections, iou_threshold=0.5)

# Visualize results
def visualize_detections(image, detections, class_names):
    """Draw bounding boxes on image"""
    image_np = cv2.imread(str(image_path))
    
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        confidence = det['confidence']
        class_name = det['class_name']
        
        # Draw bounding box
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Put label
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(image_np, label, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow('Detections', image_np)
    cv2.waitKey(0)

visualize_detections(image_path, detections, class_names)
```

### Inference Speed

```
Component           Time (ms)   % of Total
──────────────────────────────────────────
Image preprocessing    20         5%
Backbone             300        70%
Encoder              100        20%
Decoder              30         7%
Prediction Head      10         2%
Post-processing      20         5%

Total: ~430 ms (~2 FPS on GPU)

Comparison:
  YOLOv8: ~30 ms (~33 FPS)
  DETR: ~430 ms (~2 FPS)
  
DETR is ~10x slower but more accurate on small objects
```

---

## Optimization Strategies

### 1. TorchScript Export

```python
# Convert to TorchScript (faster inference)
model.eval()
scripted_model = torch.jit.script(model)
scripted_model.save('detr_scripted.pt')

# Use scripted model
loaded_model = torch.jit.load('detr_scripted.pt')
with torch.no_grad():
    outputs = loaded_model(image_tensor)
```

### 2. Quantization

```python
# Quantize model to int8 (smaller, faster)
from torch.quantization import quantize_dynamic

quantized_model = quantize_dynamic(
    model,
    {torch.nn.Linear},  # Which layers to quantize
    dtype=torch.qint8
)

# Save quantized model
torch.save(quantized_model.state_dict(), 'detr_quantized.pth')
```

### 3. ONNX Export

```python
# Export to ONNX (framework-agnostic)
import torch.onnx

dummy_input = torch.randn(1, 3, 800, 1200).to(device)

torch.onnx.export(
    model,
    dummy_input,
    "detr.onnx",
    input_names=['images'],
    output_names=['class_logits', 'bbox_preds'],
    opset_version=11
)

# Use ONNX Runtime for inference (faster)
import onnxruntime as ort

sess = ort.InferenceSession('detr.onnx')
outputs = sess.run(None, {'images': image_np})
```

### 4. Batch Inference

```python
# Process multiple images at once
batch_size = 8
batch_images = []

for i in range(batch_size):
    image = Image.open(f'image_{i}.jpg')
    image_tensor = transform(image)
    batch_images.append(image_tensor)

# Stack into batch
batch_tensor = torch.stack(batch_images).to(device)

# Single forward pass
with torch.no_grad():
    batch_outputs = model(batch_tensor)

# Process each output
for i in range(batch_size):
    detections_i = postprocess(batch_outputs, i)
```

---

## Evaluation Metrics

### COCO Metrics

```python
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Evaluate on COCO dataset
coco_gt = COCO('annotations_file.json')
results = []

for image_id, image_path in enumerate(test_images):
    detections = inference(image_path)
    
    for det in detections:
        result = {
            'image_id': image_id,
            'category_id': det['class_id'],
            'bbox': [det['x1'], det['y1'], det['w'], det['h']],
            'score': det['confidence']
        }
        results.append(result)

# Evaluate
coco_dt = coco_gt.loadRes(results)
coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# Print metrics
print(f"AP (Average Precision): {coco_eval.stats[0]:.3f}")
print(f"AP50: {coco_eval.stats[1]:.3f}")
print(f"AP75: {coco_eval.stats[2]:.3f}")
print(f"AP_small: {coco_eval.stats[3]:.3f}")
print(f"AP_medium: {coco_eval.stats[4]:.3f}")
print(f"AP_large: {coco_eval.stats[5]:.3f}")
```

### Typical Results (DETR on COCO)

```
Metric          Value
──────────────────────
AP              42.9%
AP50            60.8%
AP75            46.8%
AP_small        23.1%
AP_medium       46.2%
AP_large        59.5%

Comparison:
  Faster R-CNN: AP 42.0%
  YOLO: AP varies by version (usually 30-50%)
```

---

## Common Issues and Solutions

### Issue 1: Out of Memory
```
Solution:
  - Reduce batch size
  - Reduce image size
  - Use gradient checkpointing
  - Use mixed precision (fp16)
```

### Issue 2: Slow Training
```
Solution:
  - Use distributed training (DataParallel)
  - Use gradient accumulation
  - Reduce model size (fewer layers)
```

### Issue 3: Poor Performance on Small Objects
```
Solution:
  - Use Feature Pyramid Network (FPN)
  - Increase training data
  - Use multi-scale training
  - Consider deformable DETR
```

---

## Summary

**Training DETR:**
- ✅ Requires Hungarian matching
- ✅ Slow convergence (300+ epochs)
- ✅ Needs patience and careful tuning

**Inference with DETR:**
- ✅ Simple forward pass
- ✅ No complex post-processing
- ✅ Slower than YOLO but more accurate on small objects

**Optimization:**
- ✅ TorchScript, Quantization, ONNX
- ✅ Batch inference for speed
- ✅ Model parallelism for large models

---

## Next: Advantages and Disadvantages

→ **[Advantages and Disadvantages](./advantages-disadvantages.md)**

---

**Remember:** Training takes time, but the elegance of DETR makes it worth it! 🚀
