# 🧠 Transformer Encoder from Scratch (PyTorch)

A **from-scratch implementation** of a **Transformer Encoder** in PyTorch, including **self-attention**, **multi-head attention**, **feed-forward networks**, positional embeddings, and a **classification head** for sequence classification tasks.

This repository provides a **clean, modular, and educational implementation** of the Transformer encoder architecture, without relying on high-level abstractions such as `nn.Transformer` or external NLP frameworks.

---

## ✨ Key Features

- 🔍 **Scaled Dot-Product Attention** implemented manually  
- 🧩 **Multi-Head Attention** with head concatenation and output projection  
- 🧠 **Transformer Encoder Layers** with:
  - Layer Normalization  
  - Residual connections  
  - Feed-Forward Networks with GELU  
- 🧬 **Token + Positional Embeddings**  
- 🏷️ **Classification head** for sequence-level prediction  
- 🧪 **Unit tests with `pytest`** validating:
  - Tensor shapes and dimensions
  - Attention weight properties
  - Residual connections and normalization
  - Gradient flow and backward pass
- 🧱 Fully modular, inspectable architecture  

---

## 📁 Project Structure

```text
.
├── src/
│   ├── __init__.py
│   └── models.py              # Transformer Encoder implementation
├── tests/
│   ├── __init__.py
│   └── test_models.py         # Unit tests for encoder components
├── requirements.txt
└── README.md
```

Both `src` and `tests` are structured as **Python packages** (they include `__init__.py`), enabling clean imports and scalable organization.

---

## 🧠 Implemented Components

### Encoder Architecture
- `AttentionHead`
- `MultiHeadAttention`
- `FeedForward`
- `TransformerEncoderLayer`
- `Embeddings`
- `TransformerEncoder`

### Classification
- `ClassificationHead`
- `TransformerForSequenceClassification`

Each component closely follows the original Transformer design (*Attention Is All You Need*), with modern adaptations such as **GELU activations** and **Layer Normalization**.

---

## 🚀 Installation

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🧪 Running the Tests

All core components are covered by unit tests:

```bash
pytest
```

The test suite verifies correctness, stability, and gradient propagation.

---

## ✍️ Example Usage

```python
import torch
from src.models import TransformerForSequenceClassification

model = TransformerForSequenceClassification(
    vocab_size=100,
    max_position_embeddings=64,
    d_model=32,
    num_attention_heads=4,
    intermediate_size=64,
    num_hidden_layers=2,
    num_classes=3,
    dropout_prob=0.1,
)

input_ids = torch.randint(0, 100, (2, 10))
logits = model(input_ids)

print(logits.shape)  # (2, 3)
```

---

## 🎯 Project Goals

- Develop a strong intuition for **self-attention mechanisms**
- Provide a **readable reference implementation** of a Transformer encoder
- Serve as a foundation for:
  - Encoder-only models (BERT-style)
  - Attention visualization
  - Research and educational experimentation

---

## 📚 Technologies Used

- Python
- PyTorch
- pytest

---

## 📝 Final Notes

This repository prioritizes **clarity and correctness** over heavy optimization.  
It is designed for understanding the internal mechanics of **encoder-based Transformer models**.

✨ *Attention is all you need — understanding it is even better.*

