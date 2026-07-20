---
sidebar_position: 3
title: Object Detection
description: Concise course notes on object detection, YOLO, DETR, DINO, and Grounding DINO.
---

# Object Detection

This module introduces the core ideas behind object detection and the main model families used in modern computer vision.

## Core concept

Object detection answers two questions:
- What object is present?
- Where is it located?

The output is usually:
- a class label
- a bounding box

Common evaluation terms:
- IoU: overlap between predicted and ground-truth boxes
- Precision/Recall
- mAP: standard detection metric

## Key model notes

### YOLO
- One-stage detector designed for speed
- Predicts boxes and classes in a single pass
- Common choice for real-time systems
- Paper: https://arxiv.org/pdf/1506.02640
- Hugging Face: placeholder
- PPT: placeholder
- Notebook: placeholder

### DETR
- Transformer-based detector
- Uses object queries instead of hand-crafted anchors
- Removes the need for NMS in the standard formulation
- Paper: https://arxiv.org/pdf/2005.12872
- Hugging Face: placeholder
- PPT: placeholder
- Notebook: placeholder

### DINO
- Improves DETR with denoising and better training behavior
- Stronger convergence and more stable optimization
- Paper: https://arxiv.org/pdf/2203.03605
- Hugging Face: placeholder
- PPT: placeholder
- Notebook: placeholder

### Grounding DINO
- Open-vocabulary detector
- Connects text prompts with visual regions
- Useful for language-guided detection and zero-shot settings
- Paper: https://arxiv.org/pdf/2303.05499
- Hugging Face: placeholder
- PPT: placeholder
- Notebook: placeholder

## Shared resources
- Data (zip file): placeholder

## Suggested study flow
1. Review object detection basics
2. Study YOLO
3. Study DETR
4. Compare DINO and Grounding DINO
5. Practice with notebooks and small examples
