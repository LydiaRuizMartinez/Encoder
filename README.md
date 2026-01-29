# 🧠 Transformer Encoder from Scratch (PyTorch)

A **from-scratch implementation** of a **Transformer Encoder** in PyTorch, including **self-attention**, **multi-head attention**, **feed-forward networks**, positional embeddings, and a **classification head** for sequence classification tasks.

The goal of this repository is to provide a **clear, modular, and educational implementation** of the Transformer architecture, without relying on high-level abstractions such as `nn.Transformer` or external NLP libraries.

---

## ✨ Key Features

- 🔍 Manual implementation of **Scaled Dot-Product Attention**
- 🧩 **Multi-Head Attention** with head concatenation and output projection
- 🧠 **Transformer Encoder Layer** with:
  - Layer Normalization
  - Residual connections
  - Feed-Forward Network using GELU
- 🧬 **Token + Positional Embeddings**
- 🏷️ **End-to-end model for sequence classification**
- ✅ **Extensive unit tests with `pytest`** covering:
  - Tensor shapes
  - Attention behavior
  - Gradient flow and backward pass
  - Dropout behavior in train vs eval mode

---

## 📁 Project Structure

```text
.
├── src/
│   ├── __init__.py
│   └── models.py          # Transformer implementation
├── tests/
│   ├── __init__.py
│   └── test_models.py     # Unit tests with pytest
├── requirements.txt
└── README.md
