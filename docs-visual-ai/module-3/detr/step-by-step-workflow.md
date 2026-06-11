---
sidebar_position: 4
title: DETR Step-by-Step Workflow
description: Complete walkthrough of how DETR processes an image from input to final detections
---

# DETR Step-by-Step Workflow

## Complete Example: Detecting Objects in an Image

Let's walk through a complete example of DETR processing an image with three objects: a dog, a cat, and a person.

```
Original Image:
┌─────────────────────────────┐
│                             │
│      🐶          🐱         │
│                             │
│         👨                  │
│                             │
└─────────────────────────────┘
Size: 800 × 1200 × 3 (height × width × channels)
```

---

## Step 1: Feature Extraction (CNN Backbone)

### Input
```
Raw Image: (3, 800, 1200)
```

### Process
The ResNet-50 backbone extracts visual features through multiple convolutional blocks.

```
Input Image (800, 1200, 3)
        │
        ▼ Stage 1 (Conv 7×7, stride 2)
     (400, 600, 64)
        │
        ▼ Stage 2 (stride 2)
     (200, 300, 256)
        │
        ▼ Stage 3 (stride 2)
     (100, 150, 512)
        │
        ▼ Stage 4 (stride 2)
     (50, 75, 1024)
        │
        ▼ Stage 5 (stride 2)
     (25, 38, 2048)
        
Final stride: 32 (original / 32)
```

### Output
```
Feature Map: (2048, 25, 38)
  - Height: 25 (= 800 / 32)
  - Width: 38 (= 1200 / 32)
  - Channels: 2048
  
Contains rich information:
  ✓ Object shapes
  ✓ Textures
  ✓ Patterns
  ✓ Edges
```

### Visualization
```
Original Image:          Feature Map:
┌──────────────┐        ┌─────────────┐
│ 🐶  🐱       │        │ ▓▓▓ ▓▓▓     │
│              │        │             │
│   👨         │        │   ▓▓▓       │
└──────────────┘        └─────────────┘

High-level: 2048 channels encoding visual content
```

---

## Step 2: Flattening and Positional Encoding

### Flattening
```
Feature Map: (2048, 25, 38)
        │
        ▼ Flatten spatial dimensions
     (25*38, 2048)
     = (950, 2048)
```

### Adding Positional Encoding

**Why?** Transformers don't know spatial positions. We need to encode WHERE each feature comes from.

```
Positional Encoding (2D):

For each position (i, j) in feature map:
  PE_x[i][d] = sin(i / 10000^(d/d_model))
  PE_y[j][d] = cos(j / 10000^(d/d_model))

Concatenate: [PE_x, PE_y] dimensions

Add to features:
  Enhanced_feature = feature + positional_encoding
```

### Example
```
Feature at position (5, 10):
  Original: [0.1, -0.2, 0.05, ..., 0.3]  (2048 dims)
  
  Positional Encoding for (5, 10):
    PE: [0.8, -0.6, 0.2, ..., 0.9]  (2048 dims)
  
  Enhanced Feature:
    [0.1+0.8, -0.2-0.6, 0.05+0.2, ..., 0.3+0.9]
    = [0.9, -0.8, 0.25, ..., 1.2]

Result: Feature now knows its spatial location!
```

### Output
```
Features with Positional Encoding: (950, 2048)
  - Each element contains both:
    ✓ Visual information
    ✓ Spatial location information
```

---

## Step 3: Transformer Encoder

### Purpose
Learn **global relationships** between all features in the image.

### Input
```
Features with positions: (950, 2048)
```

### Layer Structure (6 layers)

```
Layer 1:
  Input: (950, 2048)
    │
    ├─ Multi-Head Self-Attention
    │   Feature at each position attends to ALL features
    │   Output: (950, 2048)
    │
    ├─ Add & Normalize
    │   output = LayerNorm(feature + attention_output)
    │
    ├─ Feed-Forward Network
    │   (950, 2048) → (950, 2048*4) → (950, 2048)
    │
    └─ Add & Normalize
        output = LayerNorm(FFN_output + residual)
        
  Output: (950, 2048)

Layer 2-6: Same structure, input/output (950, 2048)
```

### What Happens in Self-Attention

```
For each feature position, we compute attention:

Feature at position (5, 10) asks:
  "Which other features should I attend to?"
  
Self-attention learns:
  - Dog-related features look at other dog features (high attention)
  - Background features attend to similar background features
  - Features learn complementary information

Example attention weights for feature at (5, 10):
  Attention to (5, 10):   0.3 (self)
  Attention to (4, 10):   0.15 (nearby dog part)
  Attention to (6, 11):   0.12 (nearby dog part)
  Attention to (20, 25):  0.05 (background)
  ... (distributed over 950 positions)
```

### Example: Detecting Dog Feature

```
Dog feature at position (7, 15):

Step 1: Compute similarities with all 950 features
  Dog feature: [d1, d2, d3, ..., d2048]
  
  Compare to:
    Feature (5, 10): similarity = 0.8 (high - same dog)
    Feature (7, 14): similarity = 0.7 (high - same dog)
    Feature (20, 25): similarity = 0.1 (low - background)
    ...

Step 2: Normalize with softmax
  All similarities sum to 1.0
  
Step 3: Extract weighted values
  Output = 0.8*value(5,10) + 0.7*value(7,14) + ... + 0.1*value(20,25)

Result:
  Feature (7, 15) becomes aware of:
    - What the whole dog looks like
    - Context around the dog
    - Distinction from cat and person
```

### Output
```
Encoder Output: (950, 2048)
  - Each feature position now contains:
    ✓ Original visual information
    ✓ Positional information
    ✓ Global context (aware of entire image!)
    
Example: Dog feature now "knows"
  - Its own shape
  - There's a person nearby
  - Background characteristics
  - Differences from cat/person
```

---

## Step 4: Object Queries

### Initialize
```
Create 100 learnable object queries:

queries = [query_1, query_2, ..., query_100]

Each query: dimension (256)
  query_1: [-0.2, 0.5, -0.1, ..., 0.3]
  query_2: [0.1, -0.3, 0.2, ..., -0.4]
  ...
  query_100: [0.4, 0.2, -0.5, ..., 0.1]

Shape: (100, 256)

These are LEARNABLE - will be trained!
```

### What Do Queries Do?

```
Think of each query as asking a question:

Query 1: "Is there an object for me?" → (learns to detect)
Query 2: "Is there an object for me?" → (learns to detect)
Query 3: "Is there an object for me?" → (learns to detect)
...
Query 100: "Is there an object for me?" → (learns to detect)

During training:
  Some queries learn to detect: people
  Some queries learn to detect: dogs
  Some queries learn to detect: cats
  Some queries learn to detect: (nothing - return background)

After training:
  Queries specialize based on what objects appear in training data
```

---

## Step 5: Transformer Decoder

### Purpose
Refine object queries by attending to encoder features and other queries.

### Layer Structure (6 layers)

```
Layer 1:

Input:
  - Object Queries: (100, 256)
  - Encoder Output: (950, 2048)
  
Step 1: Self-Attention (queries attend to queries)
  Query[0] asks: "What are other queries detecting?"
  Query[1] asks: "What are other queries detecting?"
  ...
  
  Each query learns relationships with other queries
  
  Helps queries specialize:
    - Person query doesn't compete with dog query
    - Different queries learn different objects
  
  Output: (100, 256)

Step 2: Cross-Attention (queries attend to encoder output)
  Query[0] asks: "What image features match me?"
  Query[1] asks: "What image features match me?"
  ...
  
  Each query selectively attends to relevant image features
  
  Example for Dog query:
    Attends strongly to: [7,15], [8,14], [9,16] (dog features)
    Attends weakly to: [20,25], [21,26] (background)
  
  Output: (100, 256)

Step 3: Feed-Forward Network
  Process each query independently
  Add non-linearity
  
  Output: (100, 256)

Layer 2-6: Same structure, input/output (100, 256)
```

### Detailed Cross-Attention Example

```
Dog Query Cross-Attention:

Input:
  Dog Query: (256)
  Encoder Features: (950, 2048)

Compute Attention Scores:
  Q (Query): (256)
  K (Features): (950, 2048)
  
  Attention_scores = Q · K^T
                   = (256) · (2048, 950)
                   = (950)  [score for each feature]

Softmax Normalization:
  For dog query:
    Feature (7, 15) → 0.25 (high)
    Feature (8, 14) → 0.20 (high)
    Feature (9, 16) → 0.18 (high)
    Feature (20, 25) → 0.02 (low - background)
    Feature (5, 10) → 0.15 (medium - other dog)
    ... (sum = 1.0)

Extract Values:
  V (Feature values): (950, 2048)
  
  Output = softmax(scores) · V
         = 0.25*V(7,15) + 0.20*V(8,14) + 0.18*V(9,16) + ...
         = (2048)  weighted combination of features

Result:
  Dog Query becomes enriched with:
    - Primary dog features (high weights)
    - Supporting dog features
    - Some background context
    - NOT cat/person features (low weights)
```

### Multi-Layer Refinement

```
Layer 1 Output:
  Dog Query: Roughly recognizes dog features
  Cat Query: Roughly recognizes cat features
  Person Query: Roughly recognizes person features

Layer 2-3:
  Queries refine predictions
  Learn spatial relationships
  
Layer 4-5:
  Further refinement
  Handle complex interactions
  
Layer 6:
  Final refined queries
  Ready for prediction heads
```

### Output
```
Decoder Output: (100, 256)
  Each query now contains:
    ✓ Information about what it should detect
    ✓ Refined predictions based on image
    ✓ Awareness of other objects
    ✓ Spatial location information
```

---

## Step 6: Prediction Heads

### Class Prediction

```
Input: Decoder output (100, 256)

Linear Layer: (100, 256) → (100, 81)
  81 = 80 COCO classes + 1 background class

Softmax:
  Convert to probabilities
  For each query: [p_person, p_dog, p_cat, ..., p_background]

Example outputs:
  Query[0]: [0.88, 0.05, 0.02, ..., 0.05]  → Person (88%)
  Query[1]: [0.01, 0.85, 0.03, ..., 0.11]  → Dog (85%)
  Query[2]: [0.02, 0.03, 0.82, ..., 0.13]  → Cat (82%)
  Query[3]: [0.01, 0.02, 0.01, ..., 0.96]  → Background (96%)
  Query[4-100]: Similar, mostly background
```

### Bounding Box Prediction

```
Input: Decoder output (100, 256)

Linear Layer: (100, 256) → (100, 4)

Output format: [x, y, w, h]
  x, y = normalized center coordinates (0 to 1)
  w, h = normalized width and height (0 to 1)

Sigmoid activation (for x, y):
  Ensures coordinates are in [0, 1]

ReLU activation (for w, h):
  Ensures positive width/height

Example predictions:
  Query[0]: [0.25, 0.35, 0.15, 0.40]
    → Person centered at (200, 280) with size (120, 320) pixels
  
  Query[1]: [0.65, 0.40, 0.12, 0.25]
    → Dog centered at (520, 320) with size (96, 200) pixels
  
  Query[2]: [0.70, 0.55, 0.10, 0.30]
    → Cat centered at (560, 440) with size (80, 240) pixels
```

### Denormalization (Converting to Pixel Coordinates)

```
Normalized prediction: [0.25, 0.35, 0.15, 0.40]
Image size: 800 × 1200

Center x = 0.25 * 1200 = 300 pixels
Center y = 0.35 * 800 = 280 pixels
Width = 0.15 * 1200 = 180 pixels
Height = 0.40 * 800 = 320 pixels

Convert to (xmin, ymin, xmax, ymax):
  xmin = 300 - 180/2 = 210
  ymin = 280 - 320/2 = 120
  xmax = 300 + 180/2 = 390
  ymax = 280 + 320/2 = 440

Final bbox: (210, 120, 390, 440)
```

---

## Step 7: Post-Processing

### Filter by Confidence

```
Raw predictions:
  Query[0]: Person, 0.88 confidence, bbox
  Query[1]: Dog, 0.85 confidence, bbox
  Query[2]: Cat, 0.82 confidence, bbox
  Query[3]: Background, 0.96 confidence
  Query[4]: Background, 0.93 confidence
  ...
  Query[100]: Background, 0.97 confidence

Threshold: 0.5

After filtering:
  Query[0]: Person, 0.88 ✓ (keep)
  Query[1]: Dog, 0.85 ✓ (keep)
  Query[2]: Cat, 0.82 ✓ (keep)
  Query[3-100]: < 0.5 or background (remove)

Result: 3 detections
```

### Optional NMS

```
Although DETR rarely produces duplicates, we can apply light NMS:

IoU Threshold: 0.5

Remaining predictions:
  Person: (210, 120, 390, 440), 0.88
  Dog: (500, 250, 650, 500), 0.85
  Cat: (700, 350, 850, 650), 0.82

Check IoU between each pair:
  IoU(Person, Dog) = 0.05 (low - different locations)
  IoU(Person, Cat) = 0.02 (low - different locations)
  IoU(Dog, Cat) = 0.10 (low - different locations)

All IoU < threshold → Keep all detections
```

---

## Step 8: Final Detections

### Output Format

```python
[
    {
        "class": "person",
        "confidence": 0.88,
        "bbox": [210, 120, 390, 440]
    },
    {
        "class": "dog",
        "confidence": 0.85,
        "bbox": [500, 250, 650, 500]
    },
    {
        "class": "cat",
        "confidence": 0.82,
        "bbox": [700, 350, 850, 650]
    }
]
```

### Visualization

```
Original Image with Detections:
┌─────────────────────────────┐
│ dog(0.85)                   │
│ +──────+     cat(0.82)      │
│ |      |     +────────+     │
│ |      |     |        |     │
│ +──────+     |        |     │
│              +────────+     │
│              person(0.88)   │
│       +──────────────────+  │
│       |                  |  │
│       |                  |  │
│       +──────────────────+  │
└─────────────────────────────┘
```

---

## Complete Pipeline Summary

```
Image (800, 1200, 3)
        ↓ [1] Backbone
Feature Map (2048, 25, 38)
        ↓ [2] Positional Encoding
Features with Position (950, 2048)
        ↓ [3] Transformer Encoder (6 layers)
Encoded Features (950, 2048)
        │
        ├─→ [4] Object Queries (100, 256)
        │
        ↓ [5] Transformer Decoder (6 layers)
Refined Queries (100, 256)
        │
        ├─→ [6a] Class Head → (100, 81)
        │
        └─→ [6b] Bbox Head → (100, 4)
        
        ↓ [7] Post-Processing
        
        ↓
Final Detections (variable number)
[class, confidence, bbox] × N objects
```

---

## Time Complexity

```
Operation                      Complexity    Notes
─────────────────────────────────────────────────────
Backbone (ResNet-50)          O(N)          N = input pixels
Positional Encoding           O(N)
Encoder (6 layers)            O(N²)         Self-attention is quadratic
Decoder (6 layers)            O(Q·N)        Q=queries(100), N=features
Prediction Heads              O(Q)
Post-processing               O(Q²)         Optional NMS

Total: Dominated by O(N²) from encoder self-attention
```

---

## Key Insights

✅ **Backbone** extracts spatial features  
✅ **Encoder** learns global relationships  
✅ **Queries** specialize through training  
✅ **Decoder** refines query predictions  
✅ **Heads** convert to final predictions  
✅ **Post-processing** filters results  

---

## Next: Training and Inference

→ **[Training and Inference](./training-inference)

---

**Remember:** Each step builds on the previous, resulting in a complete object detection pipeline! 🚀
