---
language:
- en
license: apache-2.0
datasets:
- atrost/financial_phrasebank
- allenai/prosocial-dialog
- AI-Lab-Makerere/beans
- garage-bAInd/Open-Platypus
metrics:
- accuracy
- f1
- precision
- recall
model-index:
- name: SalesA AI
  results:
  - task:
      type: text-classification
    dataset:
      name: financial_phrasebank
      type: atrost/financial_phrasebank
    metrics:
    - type: accuracy
      value: 0.85
    - type: f1
      value: 0.83
tags:
- art
---

## **Model Name with Parameters**

**SalesA AI 125M — Multimodal Mixture-of-Experts Transformer**

- **Full Name Example:**  
  `SalesA AI 125M (MoE, Multimodal, N.E.N & SalesA Team)`

- **Breakdown:**
  - **SalesA AI**: Your model’s brand/architecture.
  - **125M**: Number of parameters (from your logs: `Total parameters: 125,106,858` ≈ 125M).
  - **MoE**: Mixture-of-Experts.
  - **Multimodal**: Handles text, vision, audio, code, action.
  - **Author**: N.E.N (Nthuku Elijah Nzeli) and SalesA Team.

---

  **Created by N.E.N (Nthuku Elijah Nzeli) and SalesA Team**

  - Model architecture: `SalesAModel`
  - Library: Custom (see included code)
  - Parameters: 125M
  - Modalities: Text, Vision, Audio, Code, Action/Robotics
  - MoE: Yes

  This repository contains the SalesA AI model, a modular, extensible, and efficient multimodal transformer with Mixture-of-Experts (MoE) layers.
  ```

---

## **Why This Naming Convention?**

- **Clarity:** Users immediately know the model’s size and capabilities.
- **Discoverability:** Easy to search/filter on Hugging Face and other platforms.
- **Professionalism:** Follows conventions used by top models (e.g., “Qwen2.5-7B-Minivoc-32k” [source](https://huggingface.co/kaitchup/Qwen2.5-7B-Minivoc-32k-v0.1a), “distilbert-base-uncased” [source](https://huggingface.co/distilbert-base-uncased)).

---

## **Summary Table**

| Field         | Value                                      |
|---------------|--------------------------------------------|
| Model Name    | SalesA AI 125M                             |
| Architecture  | Multimodal Mixture-of-Experts Transformer  |
| Parameters    | 125M                                       |
| Author        | N.E.N (Nthuku Elijah Nzeli) and SalesA Team|

---

> **SalesA AI 125M — Multimodal Mixture-of-Experts Transformer**  
> **Created by N.E.N (Nthuku Elijah Nzeli) and SalesA Team**


This repository contains the SalesA AI model, a modular, extensible, and efficient multimodal transformer with Mixture-of-Experts (MoE) layers. It supports text, vision, audio, code, and action/robotics tasks.

- Model architecture: `SalesAModel`
- Library: Custom (see included code)
- Author: N.E.N (Nthuku Elijah Nzeli) and SalesA Team

## Model Description

SalesA AI is a lightweight, CPU-optimized, multimodal model with a Mixture-of-Experts (MoE) architecture. It supports:
- **Text generation & classification** (including financial sentiment)
- **Vision (image-to-text, classification)**
- **Audio (audio-to-text, classification)**
- **Code generation**
- **Action prediction for robotics/locomotion**
- **Ethical and bias-aware outputs**

The model is designed for extensibility, ethical deployment, and real-world applications in finance, sales, stock/market analysis, and robotics.

## Intended Uses & Limitations

### Intended Uses
- Financial news and sentiment analysis
- Sales and market trend analysis
- General text, vision, and audio tasks
- Code generation and completion
- Robotics: action/command prediction from multimodal input
- Research and educational use

### Limitations
- Not suitable for high-stakes financial decisions without human oversight
- May not generalize to all languages or domains
- Biases may exist in training data; see bias analysis plots
- Not for commercial use without review (see license)

## Datasets Used
- [atrost/financial_phrasebank](https://huggingface.co/datasets/atrost/financial_phrasebank) (financial sentiment)
- [allenai/prosocial-dialog](https://huggingface.co/datasets/allenai/prosocial-dialog) (dialogue)
- [AI-Lab-Makerere/beans](https://huggingface.co/datasets/AI-Lab-Makerere/beans) (vision)
- [garage-bAInd/Open-Platypus](https://huggingface.co/datasets/garage-bAInd/Open-Platypus) (instruction following)

## Training Details
- **Architecture:** Mixture-of-Experts (MoE), multimodal encoders (text, vision, audio)
- **Parameters:** ~125M
- **Hardware:** CPU-optimized, trainable on commodity hardware
- **Losses:** Cross-entropy for classification/generation, multitask loss
- **Optimizer:** AdamW
- **Batch size:** 4 (default)
- **Epochs:** 10 (default)

## Evaluation Results
- **Financial sentiment (accuracy):** 0.85
- **Financial sentiment (F1):** 0.83
- **General text/vision/audio:** See per-task metrics in training logs
- **Bias/diagnostic plots:** See `confusion_matrix.png`, `class_distribution.png`, `per_class_metrics.png` in the model directory

## Ethical Considerations & Bias Analysis
- Model includes bias and diagnostic visualizations for transparency
- Not for use in applications requiring guaranteed fairness or absence of bias
- See [Hugging Face Model Card Guide](https://huggingface.co/docs/hub/model-cards) for best practices

## Files Included
- `model.safetensors`: Model weights
- `config.json`: Model configuration
- `tokenizer.json`, `vocab.json`, `tokenizer.model`: Tokenizer files
- `merge.txt`: Tokenizer merges (placeholder)
- `generation_config.json`: Generation parameters
- `model.safetensors.index.json`: Index for sharded weights (placeholder)
- `chat_template.jinja`: Chat UI template (placeholder)
- `training_history.pkl`: Training history
- `confusion_matrix.png`, `class_distribution.png`, `per_class_metrics.png`: Diagnostic plots

## How to Use
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# Or use your custom loading code for SalesA AI
```

## Citation
If you use this model, please cite:
```bibtex
@article{Malo2014GoodDO,
  title={Good debt or bad debt: Detecting semantic orientations in economic texts},
  author={P. Malo and A. Sinha and P. Korhonen and J. Wallenius and P. Takala},
  journal={Journal of the Association for Information Science and Technology},
  year={2014},
  volume={65}
}
```

## License
apache-2.0

## Contact
For questions or commercial licensing, contact the [SalesA Team](elijahnzeli924@gmail.com).