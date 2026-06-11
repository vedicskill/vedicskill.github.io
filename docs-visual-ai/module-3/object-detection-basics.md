---
sidebar_position: 2
title: Object Detection Basics
description: Understanding the fundamentals of object detection - bounding boxes, coordinates, and outputs
---

# Object Detection Basics

## What is Object Detection?

Object detection is the task of identifying and locating objects in images. It extends image classification by not only answering **"What is it?"** but also **"Where is it?"**

### The Fundamental Difference

#### Image Classification
```
Input: Image
Output: Class Label
```

**Example:**
```
Image: 🐶
Output: Dog
```

The model knows a dog exists but has **no information** about its location.

---

#### Object Detection
```
Input: Image
Output: Class Label + Bounding Box Location
```

**Example:**
```
Image: 🐶 🐱 👨
Output: 
  - Dog at (50, 100, 200, 300)
  - Cat at (300, 120, 450, 280)
  - Person at (100, 50, 350, 500)
```

## Why is Object Detection Important?

Consider an image with multiple objects:

```
+------------------------------+
|  Dog      Cat         Person |
|                              |
|         (scattered)          |
+------------------------------+
```

### Limitations of Image Classification

If we use image classification, we can only get **one** output:
- "Dog" ❌ What about the cat and person?
- "Cat" ❌ What about the dog and person?
- "Person" ❌ What about the dog and cat?

**Problems:**
- ❌ Cannot detect multiple objects
- ❌ Cannot locate objects
- ❌ Cannot tell which object is at which position
- ❌ Cannot count objects

### How Object Detection Solves This

```
Input Image (with multiple objects)
         ↓
     Detector
         ↓
Multiple Predictions:
  [Class] [Location] [Confidence]
  [Class] [Location] [Confidence]
  [Class] [Location] [Confidence]
         ↓
Complete Scene Understanding
```

## Bounding Boxes

A **bounding box** is a rectangular region that tightly encloses an object.

### Visualization

```
Original Image:
+-----------------------+
|                       |
|   +--------+          |
|   | Object |          |
|   +--------+          |
|                       |
+-----------------------+

Bounding Box (rectangle in red)
```

### Coordinate Representations

There are two common ways to represent bounding boxes:

#### 1. (xmin, ymin, xmax, ymax) Format
Also called **corner coordinates**

```
(xmin, ymin, xmax, ymax)
  ↓      ↓     ↓    ↓
  |      |     |    └─ Bottom-right y coordinate
  |      |     └─────── Bottom-right x coordinate  
  |      └───────────── Top-left y coordinate
  └──────────────────── Top-left x coordinate

Example: (50, 100, 200, 300)
  - Top-left corner: (50, 100)
  - Bottom-right corner: (200, 300)
```

**Visual:**
```
(50, 100)
   +─────────────+
   |             |
   |   Object    |
   |             |
   +─────────────+ (200, 300)
```

#### 2. (center_x, center_y, width, height) Format
Also called **center coordinates**

```
(center_x, center_y, width, height)
      ↓        ↓      ↓     ↓
      |        |      |     └─ Height of box
      |        |      └─────── Width of box
      |        └───────────── Y coordinate of center
      └────────────────────── X coordinate of center

Example: (125, 200, 150, 200)
  - Center: (125, 200)
  - Width: 150, Height: 200
```

**Visual:**
```
     (125, 200) - center
         •
    +─────────────+
    |             | height=200
    |   Object    |
    |             |
    +─────────────+
      width=150
```

### Converting Between Formats

**From (xmin, ymin, xmax, ymax) to (center_x, center_y, width, height):**
```python
xmin, ymin, xmax, ymax = 50, 100, 200, 300

center_x = (xmin + xmax) / 2  # (50 + 200) / 2 = 125
center_y = (ymin + ymax) / 2  # (100 + 300) / 2 = 200
width = xmax - xmin            # 200 - 50 = 150
height = ymax - ymin           # 300 - 100 = 200

# Result: (125, 200, 150, 200)
```

**From (center_x, center_y, width, height) to (xmin, ymin, xmax, ymax):**
```python
center_x, center_y, width, height = 125, 200, 150, 200

xmin = center_x - width / 2    # 125 - 75 = 50
ymin = center_y - height / 2   # 200 - 100 = 100
xmax = center_x + width / 2    # 125 + 75 = 200
ymax = center_y + height / 2   # 200 + 100 = 300

# Result: (50, 100, 200, 300)
```

## Object Detection Output

An object detector produces the following information for **each detected object**:

### 1. Class Label
The **type** or **category** of the object.

```
Examples:
- "dog"
- "cat"
- "person"
- "car"
- "bicycle"
```

### 2. Confidence Score
A **probability** indicating how confident the model is about the prediction.

```
Range: 0.0 to 1.0 (or 0% to 100%)

Examples:
- 0.95 means 95% confidence (very confident)
- 0.75 means 75% confidence (fairly confident)
- 0.45 means 45% confidence (uncertain)
```

### 3. Bounding Box
The **location** of the object in the image.

```
(50, 100, 200, 300)
or
(125, 200, 150, 200)
```

### Complete Output Example

```python
{
    "class": "dog",
    "confidence": 0.95,
    "bbox": [50, 100, 200, 300]
}
```

### Multiple Objects Output

```python
[
    {
        "class": "dog",
        "confidence": 0.95,
        "bbox": [50, 100, 200, 300]
    },
    {
        "class": "cat",
        "confidence": 0.87,
        "bbox": [300, 120, 450, 280]
    },
    {
        "class": "person",
        "confidence": 0.92,
        "bbox": [100, 50, 350, 500]
    }
]
```

## Real-World Applications

### 🚗 Autonomous Vehicles
**Objects to detect:**
- Cars
- Pedestrians
- Traffic signs
- Bicycles
- Traffic lights

**Why important:** Safety and navigation

### 🏥 Medical Imaging
**Objects to detect:**
- Tumors
- Abnormal cells
- Fractures
- Lesions

**Why important:** Early diagnosis and treatment

### 🛒 Retail Analytics
**Objects to detect:**
- Customers/People
- Shopping carts
- Products
- Checkout counters

**Why important:** Store analytics and optimization

### 🔒 Security & Surveillance
**Objects to detect:**
- People
- Vehicles
- Weapons
- Suspicious objects

**Why important:** Safety and threat detection

### 📦 Industrial Inspection
**Objects to detect:**
- Defective products
- Missing components
- Safety violations
- Equipment damage

**Why important:** Quality control

### 🎮 Augmented Reality
**Objects to detect:**
- Faces
- Hand gestures
- Furniture
- Landmarks

**Why important:** Realistic AR experiences

## Key Metrics in Object Detection

### 1. Precision
Of the objects the model detected, how many were correct?

```
Precision = Correct Detections / Total Detections Made

Example: 
If the model detected 100 objects and 95 were correct:
Precision = 95/100 = 0.95 (95%)
```

### 2. Recall
Of all the objects that actually exist, how many did the model find?

```
Recall = Correct Detections / Total Objects in Image

Example:
If there are 100 objects in image and model found 90:
Recall = 90/100 = 0.90 (90%)
```

### 3. IoU (Intersection over Union)
Measures how well the predicted bounding box matches the ground truth.

```
IoU = Area of Intersection / Area of Union

IoU = 0.5 means the bounding box is 50% accurate
IoU = 0.95 means the bounding box is 95% accurate
```

**Visualization:**
```
Ground Truth (green) vs Prediction (red):

Good Match (high IoU):
+---+
|███| 95% overlap
|███|
+---+

Poor Match (low IoU):
+---+
|▓  | 20% overlap
|  ▓|
+---+
```

## The Detection Pipeline

A typical object detection pipeline:

```
1. Input Image
        ↓
2. Feature Extraction (CNN Backbone)
        ↓
3. Detection Mechanism (YOLO/DETR/etc)
        ↓
4. Post-Processing (NMS, Filtering)
        ↓
5. Final Detections
```

## Summary

| Concept | Description |
|---------|-------------|
| **Object Detection** | Identifying AND locating objects in images |
| **Bounding Box** | Rectangle surrounding an object |
| **Class Label** | Type/category of detected object |
| **Confidence Score** | Probability of correct prediction |
| **IoU** | Accuracy of bounding box overlap |

## Next Steps

Now that you understand the basics, let's explore:

1. **[YOLO Overview](./yolo-overview)** - A brief introduction to real-time detection
2. **[DETR: Detection Transformer](./detr/introduction)** - Transformer-based detection ⭐
3. **[Grounding DINO](./grounding-dino/introduction)** - Open vocabulary detection ⭐

---

**Key Takeaway:** Object detection is about answering TWO questions simultaneously:
1. "What is in the image?" (Classification)
2. "Where is it?" (Localization)

This foundation will help you understand the more advanced transformer-based methods coming next!
