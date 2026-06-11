---
sidebar_position: 3
title: DETR Core Concepts
description: Understanding transformers, attention mechanisms, and set prediction in DETR
---

# DETR Core Concepts

## 1. Transformers Basics

### What Are Transformers?

Transformers are neural networks based on **attention mechanisms** instead of recurrence or convolution.

### Key Components

```
Transformer Block:
├─ Multi-Head Attention
├─ Feed-Forward Network
├─ Layer Normalization
└─ Residual Connections
```

### Why Transformers Work for Detection

| Feature | Benefit |
|---------|---------|
| **Attention** | Flexible relationships (not fixed like CNN) |
| **Global Context** | Every element sees every other element |
| **Parallelizable** | Efficient computation |
| **Extensible** | Can be applied to many tasks |

---

## 2. Attention Mechanism

### What is Attention?

Attention allows the model to focus on relevant parts of the input.

**Analogy:**
```
You're reading this text.
Your attention focuses on important words.
You don't attend equally to all words.

Similarly, attention weights determine:
  How much to focus on each part
  What to ignore
```

### Attention Formula

The **Scaled Dot-Product Attention** formula:

```
Attention(Q, K, V) = softmax((Q·K^T) / √d_k) · V

where:
  Q = Query vector (what are we looking for?)
  K = Key vector (what do we have?)
  V = Value vector (what information to extract?)
  d_k = dimension of key
```

### Attention Visualization

```
Query attends to all Keys:

Query: "Where are dogs?"
       ↓
Keys:  [dog] [sky] [tree] [dog] [grass]
Attention weights: [0.6] [0.1] [0.05] [0.2] [0.05]
       ↓
High attention on dog features
Low attention on sky/tree features
       ↓
Extract Value information from high-attention positions
```

### Step-by-Step Example

```
Suppose we have 3 elements in a sequence:

Queries (Q):        Dimensions: 3×d_model
  [q1]
  [q2]
  [q3]

Keys (K):           Dimensions: 3×d_model
  [k1]
  [k2]
  [k3]

Values (V):         Dimensions: 3×d_model
  [v1]
  [v2]
  [v3]

Step 1: Compute attention scores
  Q·K^T = 
  [q1·k1  q1·k2  q1·k3]    [score11 score12 score13]
  [q2·k1  q2·k2  q2·k3]  = [score21 score22 score23]
  [q3·k1  q3·k2  q3·k3]    [score31 score32 score33]

Step 2: Scale and normalize
  Divide by √d_model (e.g., √256 ≈ 16)
  Apply softmax to get probabilities

Step 3: Apply to values
  Attention = softmax(scores) · V
  
  Each element becomes weighted combination of all values
```

---

## 3. Multi-Head Attention

### Why Multiple Heads?

Single attention head focuses on ONE type of relationship.

**Solution:** Use multiple attention heads to capture different types of relationships.

### How Multi-Head Attention Works

```
Input: (seq_len, d_model)

Split into multiple heads:
  Head 1: (seq_len, d_model/num_heads)
  Head 2: (seq_len, d_model/num_heads)
  ...
  Head 8: (seq_len, d_model/num_heads)

Each head computes attention independently:
  Head 1: Attention1(Q1, K1, V1)
  Head 2: Attention2(Q2, K2, V2)
  ...
  Head 8: Attention8(Q8, K8, V8)

Concatenate outputs:
  Concat[output1, output2, ..., output8]

Linear projection:
  Output: (seq_len, d_model)
```

### Example with 8 Heads

```
DETR uses 8 attention heads (default)

Head 1: Focuses on object boundaries
Head 2: Focuses on texture details
Head 3: Focuses on object centers
Head 4: Focuses on color patterns
Head 5: Focuses on spatial relationships
Head 6: Focuses on edges
Head 7: Focuses on symmetry
Head 8: Focuses on context

Together: Comprehensive understanding!
```

### Visualization

```
Multi-Head Attention:

Input (4, 256):
  element1
  element2
  element3
  element4

            Split into 8 heads
            /      |      \
        Head1    Head2   ...  Head8
        Attn1    Attn2       Attn8
            \      |      /
            
           Concatenate
             |
           Linear
             |
        Output (4, 256)
```

---

## 4. Self-Attention vs Cross-Attention

### Self-Attention

**Definition:** An element attends to itself and other elements in the **same sequence**.

```
In DETR Encoder:
  Features attend to all features
  "What features in the image should I focus on?"

In DETR Decoder:
  Queries attend to queries
  "What other objects should I be aware of?"
```

**Example:**
```
Sequence: [dog] [cat] [person] [tree]

Dog attends to:
  itself [dog]
  other elements: [cat], [person], [tree]
  
Learns: "I'm a dog. Near me are cat, person, tree."
```

### Cross-Attention

**Definition:** Elements from one sequence attend to elements in a **different sequence**.

```
In DETR Decoder:
  Queries (objects to detect) attend to encoder features (image)
  "What image features are relevant to me?"
```

**Example:**
```
Query "person" attends to image features:

Person query looks at all image features
  Attends strongly to human-shaped features
  Ignores dog/cat/tree-shaped features
  
Learns: "Here are the features that look like a person"
```

### Side-by-Side Comparison

```
Self-Attention (Encoder):
  Q, K, V from same source (image features)
  Features ↔ Features
  
  Formula: Attention(F, F, F)
  where F = features

Cross-Attention (Decoder):
  Q from different source than K, V
  Queries ↔ Features
  
  Formula: Attention(Q, F, F)
  where Q = queries, F = features

Visual:
  Self-Attention:     Cross-Attention:
  ┌─────┐            ┌───────┐
  │ Feature ↔──→ ├─→ │ Query  │
  │ Feature ├───┘│   └───────┘
  │ Feature ├─────┘   Attends to features
  └─────┘    
  Features attend 
  to each other
```

---

## 5. Feed-Forward Network

### What is It?

A simple multi-layer perceptron applied to each position independently.

### Structure

```
Position-wise Feed-Forward Network:

Input: (seq_len, d_model)
       │
       ▼
  Linear Layer 1
  (d_model → d_ff)
  where d_ff = 4 × d_model (e.g., 1024)
       │
       ▼
  ReLU Activation
       │
       ▼
  Linear Layer 2
  (d_ff → d_model)
       │
       ▼
Output: (seq_len, d_model)

Same shape in, same shape out!
```

### Why Position-Wise?

```
Each element is processed independently:
  
For element at position 1:
  FFN([element1]) → [output1]

For element at position 2:
  FFN([element2]) → [output2]

All use the same FFN parameters
But applied position-by-position
```

### Example

```
Input: [[1, 2], [3, 4]]  (2 elements, 2 dimensions)

FFN1 (2 → 8):
  [[1, 2]] → [[a, b, c, d, e, f, g, h]]
  [[3, 4]] → [[i, j, k, l, m, n, o, p]]

ReLU:
  [[a, b, c, d, e, f, g, h]]  (keep positive values)
  [[i, j, k, l, m, n, o, p]]

FFN2 (8 → 2):
  [[...8 values...]] → [[1', 2']]
  [[...8 values...]] → [[3', 4']]

Output: [[1', 2'], [3', 4']]  (same shape as input)
```

---

## 6. Residual Connections

### What Are They?

A shortcut that adds the input directly to the output.

### Why Are They Important?

```
Without residual connection:
  Input ──→ [Transformation] ──→ Output

  Problem: Deep networks suffer from vanishing gradients

With residual connection:
  Input ──→ [Transformation] ──→ Addition ──→ Output
       └─────────────────────┬────────────────┘
             Shortcut (skip connection)

  Output = Transformation(Input) + Input
```

### In Transformers

```
Transformer Block:

Input ──→ Multi-Head Attention ──→ Add Input ──→ Norm
          ↓
       Output1 = Attention(Input) + Input

Output1 ──→ Feed-Forward ──→ Add Output1 ──→ Norm
            ↓
         Output2 = FFN(Output1) + Output1

Final Output: Output2
```

### Benefit

```
Gradient can flow:
  ✓ Through transformations
  ✓ Directly via shortcut

Allows:
  ✓ Deeper networks
  ✓ Faster training
  ✓ Better convergence
```

---

## 7. Set Prediction Problem

### What is Set Prediction?

Instead of predicting ordered sequences, predict **unordered sets** of objects.

### Traditional vs DETR

```
Traditional Detection (YOLO):
  Grid cells are ordered by position
  Predict: "Grid cell (i,j) contains object X"

DETR (Set Prediction):
  Objects are unordered
  Predict: "Image contains {object1, object2, ...}"
  
  Order doesn't matter:
    {dog, cat, person} = {person, dog, cat}
```

### Why This Matters

```
Traditional problem: "Which grid cell has what?"
  ├─ Objects bound to specific locations
  ├─ Forced into grid structure
  └─ Unnatural constraint

Set problem: "What objects exist?"
  ├─ Objects are just set elements
  ├─ Natural formulation
  └─ More flexible
```

### DETR Formulation

```
Given image I:
  Predict set S = {o1, o2, ..., oN}
  
  where oi = (classi, bboxi)
  
The set is:
  ✓ Unordered
  ✓ Variable size (0 to N objects)
  ✓ Predicted by 100 queries (some return "no object")
```

---

## 8. Hungarian Matching

### The Training Problem

During training, how do we match predictions to ground truth?

```
Predicted: Query 1 → Person at (100, 200)
           Query 2 → Car at (300, 400)
           Query 3 → Dog at (500, 100)

Ground truth: Person at (95, 205)
              Dog at (505, 105)
              Car at (295, 395)

Matching question:
  Which prediction corresponds to which ground truth?
  Query 1 → Person (GT 0)? ✓
  Query 2 → Car (GT 2)? ✓
  Query 3 → Dog (GT 1)? ✓
```

### The Hungarian Algorithm

An algorithm to find **optimal one-to-one matching**.

```
Given:
  - Predicted objects: {P1, P2, ..., P100}
  - Ground truth objects: {G1, G2, ..., GN}

Find:
  - Best matching between predictions and ground truth
  - Minimize total cost

Cost = Sum of (mismatch cost for each pair)

where mismatch cost includes:
  - Class mismatch
  - Bounding box IoU mismatch
```

### Example

```
Predictions: [Person(0.9), Car(0.8), Dog(0.7), NoObj, ...]
Ground truth: [Person, Dog, Car]

Hungarian algorithm finds best matching:
  Pred[0] Person ↔ GT[0] Person ✓ (match)
  Pred[1] Car ↔ GT[2] Car ✓ (match)
  Pred[2] Dog ↔ GT[1] Dog ✓ (match)
  Pred[3:] NoObj ↔ Nothing ✓ (unmatched predictions)

Loss = Sum of losses for matched pairs
```

### Why It's Needed

```
Without matching:
  "Which prediction should be compared to which ground truth?"
  
With Hungarian matching:
  "Find optimal one-to-one assignment"
  Avoid counting same object multiple times
  Ensure fair comparison during training
```

---

## 9. Loss Function

### Components

DETR loss has two parts:

```
Total Loss = Classification Loss + Localization Loss

Classification Loss:
  Measures class prediction accuracy
  Uses Cross-Entropy
  Includes background class

Localization Loss:
  Measures bounding box accuracy
  Uses L1 Loss + GIoU Loss
  
  L1 Loss: |predicted_bbox - true_bbox|
  GIoU Loss: 1 - GIoU(predicted, true)
```

### Formula

```
Loss = λ_ce * CE_loss + λ_bbox * L1_loss + λ_giou * GIoU_loss

where:
  λ_ce = weight for classification (e.g., 1.0)
  λ_bbox = weight for L1 (e.g., 5.0)
  λ_giou = weight for GIoU (e.g., 2.0)

Different weights for different losses
L1 and GIoU emphasize localization more
```

### IoU (Intersection over Union)

```
IoU = Area(A ∩ B) / Area(A ∪ B)

Prediction (A):    Ground Truth (B):    Intersection:
  +-----+          +-----+               +--+
  |     |          |     |               |  |
  +-----+          +-----+               +--+

IoU = Area(intersection) / Area(union)
```

### GIoU (Generalized IoU)

```
GIoU = IoU - (Area(C) - Area(A∪B)) / Area(C)

where C is the smallest rectangle containing both boxes

Benefits:
  ✓ Handles non-overlapping boxes
  ✓ Encourages rotation/size corrections
  ✓ Better gradient signal than IoU
```

---

## Summary of Core Concepts

| Concept | Purpose |
|---------|---------|
| **Attention** | Focus on relevant parts |
| **Multi-Head** | Capture multiple types of relationships |
| **Self-Attention** | Learn relationships within sequence |
| **Cross-Attention** | Learn relationships between sequences |
| **Feed-Forward** | Add non-linearity, capacity |
| **Residuals** | Enable deep networks |
| **Set Prediction** | Natural problem formulation |
| **Hungarian Matching** | Match predictions to ground truth |
| **Loss Function** | Guide learning process |

---

## Next: Step-by-Step Workflow

Ready to see how all these concepts come together?

→ **[Step-by-Step Workflow](./step-by-step-workflow)**

---

**Key Takeaway:** Understanding these core concepts is essential to grasp how DETR revolutionizes object detection! 🚀
