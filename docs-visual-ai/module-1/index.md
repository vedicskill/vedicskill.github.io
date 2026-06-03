---
sidebar_position: 1
---

# Big Picture

## Overview

Welcome to the **Big Picture** module.

In this module, we will explore the overall roadmap of the course and understand what we are going to build throughout this journey.

Modern AI systems are rapidly transforming the field of computer vision using **Transformers** and **Generative AI**. Traditional computer vision systems were mainly limited to detecting predefined objects. Today, modern AI models can:

- Understand images together with language
- Detect and segment objects using natural language prompts
- Analyze images and provide reasoning
- Understand videos and visual scenes
- Generate and edit images using AI
- Build multimodal AI applications that combine vision and language

This course is designed in a **practical**, **hands-on**, and **beginner-friendly** manner.

Instead of focusing only on theory, we will gradually build modern AI vision applications step by step throughout the course.


## What We Will Learn

Throughout this course, we will work with modern AI and transformer-based vision models including:

- Vision Transformers (ViT)
- CLIP
- DINO
- Grounding DINO
- Segment Anything Model (SAM)
- Stable Diffusion
- Video Transformers
- Streamlit AI Applications

We will also build real-world AI applications such as:

- Image understanding systems
- AI-powered image search engines
- Image segmentation applications
- Text-to-image generation systems
- Multimodal AI demos
- Vision-language AI applications


## Demo

In the first part of this module, we will showcase demo applications that you will eventually build by the end of the course.

These demos are designed to give you a high-level understanding of the capabilities of modern Visual AI systems.

The demonstrations will showcase AI capabilities such as:

- Image understanding
- AI image generation
- Object grounding
- Image segmentation
- Vision-language AI systems
- Interactive AI applications

> **Note:**  
> Some demo applications will be developed later in the course once we complete the required foundational modules and application development sections.


## Environment Setup

Before building advanced AI applications, we first need to prepare our AI development environment.

In this section, we will complete the full environment setup required for the course.


### 1. Install Python and VS Code

Install the following tools:

- **[Python](https://www.python.org/downloads/)**
    - Recommended verison Python > 3.11
    - Avoid
        - Python latest version
        - Ecosystem instablility with latest release
- **[Visual Studio Code (VS Code)](https://code.visualstudio.com/Download)**
      - Recommeded to install latest version
      - Extentions: Author Preferrable **Microsoft**
          - Python
          - Jupyter
          - Pylance


These tools will be used throughout the course for AI development, experimentation, and application building.


### 2. Download the Course Folder Structure from GitHub

Download the course repository and project structure from GitHub.

- Repository: [https://github.com/vedicskill/course-visual-ai-with-transformers-llms/](https://github.com/vedicskill/course-visual-ai-with-transformers-llms/)

- Default Branch: `main`

The repository contains:

- source code
- notebooks
- applications
- experiments
- datasets
- assets
- outputs


### 3. Install Poetry & Create Virtual Environment

We will use Poetry for:

- dependency management
- package installation
- virtual environment management

This helps us maintain clean, isolated, and reproducible AI development environments.


#### Install Poetry

Go to the official Poetry documentation:

- [https://python-poetry.org/docs/](https://python-poetry.org/docs/)

Navigate to the installation section and choose your preferred installation method.

The most recommended approach is using the official installer.



##### Linux, macOS, Windows (WSL)

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

##### Windows (PowerShell)

```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

Once Poetry is installed, follow the steps provided in the official documentation to initialize and manage the project environment.


#### Install Libraries

To install packages using Poetry, use the `add` command.

Example:

```bash
poetry add opencv-python
```


#### Install Required Libraries

For this course, the required libraries are already defined inside the `requirements.txt` file.

```txt
numpy
opencv-python
ultralytics
torch
torchvision
transformers
timm
accelerate
diffusers
streamlit
```

To install all required packages:

##### Linux, macOS, Windows (WSL)

```bash
poetry add $(cat requirements.txt)
```

##### Windows (PowerShell)

```powershell
Get-Content requirements.txt | ForEach-Object { poetry add $_ }
```



### 4. Quick Check on CPU vs GPU

We will briefly understand:

- What is a CPU
- What is a GPU
- Why GPUs are important for AI
- How to check GPU availability in PyTorch

This section is designed as a beginner-friendly introduction.

Advanced GPU optimization and performance tuning topics will be covered later in the course.


### 5. Run Our First AI Model

Finally, we will run our first transformer-based AI model to verify that our environment is correctly configured.

This step confirms that:

- Python is working correctly
- required libraries are installed
- PyTorch is functioning properly
- the AI environment is fully ready for the upcoming modules

#### Optional : [Configure Hugging Face Model Storage Directory](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables)

By default, when you download models using the Transformers library, the models are stored inside the system cache directory.
To keep your project organized, you can configure a dedicated folder to store all [Hugging Face models](https://huggingface.co/models) locally within the project.

Add the following code at the beginning of your Python file:
```python
import os
# add HF_HOME, TRANSFORMERS_CACHE enviroment variable and set the desired path
hf_path = "<full path of desired directory>"
os.environ["HF_HOME"] = hf_path
os.environ["TRANSFORMERS_CACHE"] = hf_path
```
This configuration ensures that all downloaded Hugging Face models are stored inside the specified `hf_path` directory instead of the default system cache location.


## Next Step

In the next lesson, we will officially begin our journey into **Visual AI with Transformers and Large Language Models (LLMs)**.
