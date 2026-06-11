---
sidebar_position: 2
title: DETR Architecture Overview
description: Detailed breakdown of DETR's architecture components
---

# DETR Architecture Overview

## Complete DETR Architecture

```
                          ┌──────────────────────────┐
                          │   Input Image (H×W×3)    │
                          └──────────┬───────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │   CNN Backbone (ResNet-50)      │
                    │  Extract Features               │
                    │  Output: (C, H/32, W/32)        │
                    └────────────────┬────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │  Positional Encoding            │
                    │  Flatten & add position info    │
                    │  Output: (N, C)                 │
                    │  where N = HW/1024              │
                    └────────────────┬────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │ Transformer Encoder             │
                    │  Multi-head attention           │
                    │  Feed-forward networks          │
                    │  Layer normalization            │
                    │  Output: (N, d_model)           │
                    └────────────────┬────────────────┘
                                     │
    ┌────────────────────────────────┼────────────────────────────────┐
    │                                │                                │
    │                                │                                │
    │         ┌──────────────────────▼──────────────────────┐         │
    │         │ Object Queries (Learnable)                  │         │
    │         │ (100, d_model)                              │         │
    │         │ - Query 1, Query 2, ..., Query 100          │         │
    │         └──────────────────────┬──────────────────────┘         │
    │                                │                                │
    │         ┌──────────────────────▼──────────────────────┐         │
    │         │ Transformer Decoder                          │         │
    │         │ - Cross-attention to encoder output          │         │
    │         │ - Self-attention between queries             │         │
    │         │ Output: (100, d_model)                       │         │
    │         └──────────────────────┬──────────────────────┘         │
    │                                │                                │
    │         ┌──────────────────────▼──────────────────────┐         │
    │         │ Prediction Head                              │         │
    │         │ ├─ Linear: (100, d_model) → (100, C+1)      │         │
    │         │ │  (C classes + 1 background)               │         │
    │         │ ├─ Linear: (100, d_model) → (100, 4)        │         │
    │         │ │  (Bounding box: x, y, w, h)               │         │
    │         │ └─ Output: class_logits, bbox_preds         │         │
    │         └──────────────────────┬──────────────────────┘         │
    │                                │                                │
    └────────────────────────────────┼────────────────────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │ Post-Processing                 │
                    │ ├─ Softmax on class logits      │
                    │ ├─ Filter by confidence         │
                    │ └─ Optional NMS                 │
                    └────────────────┬────────────────┘
                                     │
                          ┌──────────▼───────────┐
                          │ Final Detections     │
                          │ [class, conf, bbox]  │
                          └──────────────────────┘
```

## Component 1: CNN Backbone

### Purpose
Extract visual features from the input image.

### Common Backbone
**ResNet-50** is typically used.

### Process
```
Input: Image (3, H, W)
  e.g., (3, 800, 1200)
       ↓
ResNet-50 Layers:
  └─ Conv + BatchNorm + ReLU (multiple blocks)
    └─ Downsampling occurs at stages
       ↓
Output: Feature Map (C, H/32, W/32)
  e.g., (2048, 25, 38)
```

### Why ResNet-50?
- ✅ Well-pretrained on ImageNet
- ✅ Good balance of speed and accuracy
- ✅ Proven backbone architecture
- ✅ Can be easily replaced with other backbones

### Feature Map Resolution
```
Original Image:    800 × 1200
After 5 stages:    800/32 × 1200/32 = 25 × 38

Stride = 32 (typical for ResNet)
Resolution reduced 32x in each dimension
Channels: 2048 (deep feature representation)
```

---

## Component 2: Positional Encoding

### Purpose
Add spatial information to features since transformers don't have inherent spatial awareness.

### Why Needed?
Transformers operate on sequences without positional information. To tell where features come from in the image, we need positional encoding.

### Types of Positional Encoding

#### Absolute Positional Encoding
```
Feature position: (i, j) in the feature map
Encoding: sine/cosine functions of different frequencies

Formula:
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

where:
  pos = position in sequence
  i = dimension index
  d = model dimension
```

#### Spatial Positional Encoding
```
DETR typically uses 2D positional encoding:

For each (x, y) coordinate in feature map:
  - Encode x position: PE_x(x)
  - Encode y position: PE_y(y)
  - Concatenate: [PE_x(x), PE_y(y)]
```

### Example
```
Feature Map: 25 × 38 (after backbone)

Flattened: 25*38 = 950 elements

Each element (pixel) gets a positional encoding:
  Position (0,0) → [PE_x, PE_y]
  Position (0,1) → [PE_x, PE_y]
  ...
  Position (24,37) → [PE_x, PE_y]

Then added to feature values:
  Feature + PositionalEncoding = Enhanced Feature
```

---

## Component 3: Transformer Encoder

### Purpose
Learn **global relationships** and context in the image.

### Architecture
```
Input: Features with positional encoding (N, d_model)
where N = 950 (flattened feature map)
      d_model = 256 (hidden dimension)
       ↓
MultiHead Self-Attention:
  └─ Each feature attends to all other features
  └─ Learn relationships
  └─ Multiple attention heads capture different relationships
       ↓
Feed-Forward Network:
  └─ Position-wise dense layers
       ↓
Layer Normalization & Residual Connections
       ↓
Output: (N, d_model) - same shape as input
```

### Self-Attention Mechanism
```
What each feature "looks at":
  Feature at position i attends to:
    Feature at position 1
    Feature at position 2
    ...
    Feature at position N

This creates a global view of the entire image!
```

### Example
```
Image with dog at position (10, 15):

Feature at (10, 15) asks:
  "What other features should I attend to?"

Self-attention learns to attend to:
  - Other dog features
  - Background features
  - All surrounding context

Result: Feature becomes aware of global context
```

### Why This Matters
```
YOLO: Each cell only sees local neighborhood
DETR Encoder: Each feature sees entire image

This is why DETR has stronger global context!
```

---

## Component 4: Object Queries

### The Revolutionary Idea

Instead of using **anchor boxes** or **grid cells**, DETR uses **learnable object queries**.

### What Are Object Queries?

```
Learnable embeddings (vectors) that ask questions about the image.

Number of queries: 100 (fixed, configurable)
Dimension: d_model = 256

Each query:
  └─ A learnable vector
  └─ Initialized randomly at start
  └─ Learned during training
  └─ Asks: "Is there an object I should detect?"
```

### Visual Explanation

```
Imagine 100 "detectors" (queries):

Query 1: Looking for people
Query 2: Looking for dogs
Query 3: Looking for cars
...
Query 100: Looking for anything

Each query attends to encoder output and either:
  ✓ Finds something → Predicts class & bbox
  ✗ Finds nothing → Predicts "no object"
```

### How Object Queries Work

```
Initialize 100 learnable vectors:
  queries[0] = [0.1, -0.2, 0.05, ...]
  queries[1] = [0.3, 0.1, -0.1, ...]
  ...
  queries[99] = [-0.2, 0.4, 0.15, ...]

During training:
  Each query learns what to look for
  Shape of vectors changes to match objects

After training:
  Query vectors contain implicit knowledge:
    "This position should detect people"
    "This position should detect cars"
    etc.
```

### Queries vs Anchors

| Aspect | Anchors (YOLO) | Queries (DETR) |
|--------|---|---|
| **Design** | Hand-crafted | Learnable |
| **Flexibility** | Fixed sizes/ratios | Adaptive |
| **Number** | Many (millions) | Fixed (100) |
| **Learning** | Not learned | Learned from data |
| **Interpretability** | Grid positions | Object representations |

---

## Component 5: Transformer Decoder

### Purpose
Generate final object predictions by attending to:
1. **Encoder output** (image features)
2. **Object queries** (what to detect)

### Architecture

```
Object Queries (100, d_model)
        │
        ▼
    ┌──────────────────────────────┐
    │ Self-Attention               │
    │ Queries attend to each other │
    └────────┬─────────────────────┘
             │
             ▼
    ┌──────────────────────────────┐
    │ Cross-Attention              │
    │ Queries attend to features   │
    │ (Encoder output)             │
    └────────┬─────────────────────┘
             │
             ▼
    ┌──────────────────────────────┐
    │ Feed-Forward Network         │
    │ Per-query transformations    │
    └────────┬─────────────────────┘
             │
             ▼
Output: (100, d_model)
```

### Self-Attention in Decoder
```
Each query attends to other queries:

Query 1 asks: "What are other queries detecting?"
Query 2 asks: "What are other queries detecting?"
...

This helps queries specialize:
  - Query detecting people doesn't compete with cars
  - Queries learn complementary features
```

### Cross-Attention in Decoder
```
Each query attends to encoder features:

Query 1 looks at ALL image features
Query 2 looks at ALL image features
...
Query 100 looks at ALL image features

Each query "picks" which features are relevant:
  - Query for "person" attends to person-like features
  - Query for "dog" attends to dog-like features
```

### Example Workflow

```
Input:
  - Encoder output: (950, 256) - image features
  - Object queries: (100, 256) - things to detect

Decoder Processing:

Step 1: Self-Attention
  Queries attend to queries
  Result: (100, 256)
  
Step 2: Cross-Attention
  Queries attend to encoder features
  Result: (100, 256)
  
Step 3: Feed-Forward
  Process each query independently
  Result: (100, 256)
  
Multiple layers (6 by default):
  Repeat steps 1-3 six times
  Queries refine predictions iteratively
```

---

## Component 6: Prediction Head

### Purpose
Convert decoder output to final predictions.

### Structure

```
Decoder Output: (100, d_model)
        │
        ├─────────────────────────┬─────────────────────────┐
        │                         │                         │
        ▼                         ▼                         ▼
    Linear Layer            Linear Layer              
    (100, C+1)              (100, 4)
    [Class Logits]          [Bbox Coordinates]
        │                         │
        ├─────────────────────────┴─────────────────────────┤
        │                                                     │
        ▼                                                     ▼
    Softmax                                           Sigmoid/Linear
    [Class Probabilities]                            [Normalized Bbox]
        │                                                     │
        └─────────────────────────┬─────────────────────────┘
                                  │
                                  ▼
                    Final Prediction for each query:
                    [class_probability, confidence, bbox]
```

### Class Prediction
```
Input: (100, d_model)
Linear layer: (100, d_model) → (100, C+1)
Output: Logits for each class

C = number of object classes (e.g., 80 for COCO)
+1 = background class

For each of 100 queries:
  [prob_person, prob_dog, prob_car, ..., prob_background]
```

### Bounding Box Prediction
```
Input: (100, d_model)
Linear layer: (100, d_model) → (100, 4)
Output: Bounding box parameters

Parametrization:
  [center_x, center_y, width, height]
  
Normalized coordinates:
  center_x, center_y ∈ [0, 1] (relative to image)
  width, height ∈ [0, 1]
  
Formula:
  bbox = sigmoid([x, y]) for normalization
  bbox = relu([w, h]) for width/height
```

---

## Component 7: Post-Processing

### Simple Post-Processing

```
Raw predictions: (100, C+1) classes + (100, 4) bboxes
        │
        ▼
Apply Softmax:
  Get class probabilities
        │
        ▼
Filter by Confidence:
  Keep detections > threshold (e.g., 0.5)
  Remove background detections
        │
        ▼
Optional NMS:
  Remove duplicate detections
  (Usually not needed for DETR!)
        │
        ▼
Final Detections:
  [class, confidence, bbox] × N objects
```

### Why DETR Doesn't Need NMS

```
YOLO produces multiple boxes per grid cell
→ Need NMS to remove duplicates

DETR produces exactly 1 box per query
→ Rarely produces duplicates
→ NMS often unnecessary!

Some implementations still use light NMS for safety.
```

---

## Key Design Choices

| Component | Choice | Reason |
|-----------|--------|--------|
| **Backbone** | ResNet-50 | ImageNet pretrained |
| **Encoder Layers** | 6 | Balance efficiency/power |
| **Decoder Layers** | 6 | Matches encoder |
| **Hidden Dimension** | 256 | Balance speed/accuracy |
| **Attention Heads** | 8 | Multihead attention |
| **Object Queries** | 100 | Covers most scenes |
| **Positional Encoding** | 2D Sine | Generalizes well |

---

## Architecture Variations

### Modified Backbones
```
ResNet-50 (default)  →  ResNet-101
  More capacity, slower

ResNet-50  →  EfficientNet
  Better speed/accuracy trade-off
```

### Encoder/Decoder Depth
```
Default: 6 layers each

Lightweight: 3 layers each
  Faster, but less accurate

Heavy: 12 layers each
  More accurate, slower convergence
```

### Number of Queries
```
Default: 100

Few objects: 50
  Faster, sufficient for simple scenes

Many objects: 300
  Handles crowded scenes
```

---

## Complete Forward Pass Example

```
Input Image: (3, 800, 1200)

Step 1: Backbone
  → (2048, 25, 38)

Step 2: Positional Encoding
  → (950, 256) with position info

Step 3: Transformer Encoder (6 layers)
  → (950, 256) with global context

Step 4: Object Queries
  → (100, 256) learnable

Step 5: Transformer Decoder (6 layers)
  ├─ Self-Attention (queries ↔ queries)
  ├─ Cross-Attention (queries ↔ features)
  └─ Feed-Forward
  → (100, 256) refined predictions

Step 6: Prediction Head
  ├─ Class head: (100, 256) → (100, 81)
  └─ Bbox head: (100, 256) → (100, 4)

Step 7: Post-Processing
  → Final detections [class, conf, bbox]
```

---

## Summary

**DETR's architecture is elegant:**

1. **Backbone** extracts features
2. **Encoder** learns global relationships
3. **Queries** represent objects to detect
4. **Decoder** attends to features for each object
5. **Head** predicts classes and boxes
6. **Post-processing** filters results

No anchors. No NMS. No hand-crafted components.

→ **[Next: Core Concepts](./core-concepts)**
