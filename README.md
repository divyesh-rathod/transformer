# Transformer from Scratch: English to Italian Neural Machine Translation

## 🎯 Motivation

I built this project to gain a deeper understanding of Large Language Models (LLMs) and systems like ChatGPT. Rather than treating transformers as a black box, I wanted to understand the foundational architecture that powers modern AI. To achieve this, I reverse-engineered the groundbreaking paper **"Attention Is All You Need"** (Vaswani et al., 2017) and implemented a complete transformer model from scratch using PyTorch.

This project demonstrates a practical application: an **English-to-Italian translator** trained on the OPUS Books dataset.

---

## 📚 What I Built

This is a **full implementation of the Transformer architecture** for neural machine translation, including:

- ✅ Multi-head self-attention mechanism
- ✅ Positional encoding
- ✅ Encoder-decoder architecture with 6 layers each
- ✅ Feed-forward networks with residual connections
- ✅ Layer normalization
- ✅ Custom dataset handling for bilingual translation
- ✅ Training pipeline with checkpoint management
- ✅ TensorBoard integration for monitoring

---

## 🏗️ Architecture Overview

The transformer consists of the following key components:

### **1. Input Embedding (`inputEmbedding`)**

Converts token IDs into dense vector representations and scales them by √d_model for stable training.

### **2. Positional Encoding (`PositionalEncoding`)**

Adds positional information to embeddings using sine and cosine functions, allowing the model to understand word order without recurrence.

### **3. Multi-Head Attention (`MultiHeadAttentionBlock`)**

Implements the scaled dot-product attention mechanism with multiple attention heads (h=8), allowing the model to attend to different representation subspaces simultaneously.

**Attention formula:**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### **4. Feed-Forward Network (`FeedForwardBlock`)**

Two-layer fully connected network with ReLU activation:

- Layer 1: d_model (512) → d_ff (2048)
- Layer 2: d_ff (2048) → d_model (512)

### **5. Encoder (`Encoder`)**

Stacks 6 identical encoder blocks, each containing:

- Multi-head self-attention
- Feed-forward network
- Residual connections and layer normalization

### **6. Decoder (`Decoder`)**

Stacks 6 identical decoder blocks, each containing:

- Masked multi-head self-attention
- Cross-attention with encoder output
- Feed-forward network
- Residual connections and layer normalization

### **7. Projection Layer (`ProjectionLayer`)**

Maps decoder output to vocabulary size and applies log-softmax for probability distribution over target vocabulary.

---

## 📁 Project Structure

```
transformer/
├── model.py              # Complete transformer architecture
├── train.py              # Training loop and model initialization
├── dataset.py            # BilingualDataset and data preprocessing
├── config.py             # Hyperparameters and configuration
├── tokenizer_en.json     # English tokenizer vocabulary
├── tokenizer_it.json     # Italian tokenizer vocabulary
├── opus_books_weights/   # Saved model checkpoints
└── runs/                 # TensorBoard logs
```

---

## ⚙️ Configuration

Key hyperparameters (defined in `config.py`):

| Parameter    | Value | Description                               |
| ------------ | ----- | ----------------------------------------- |
| `d_model`    | 512   | Dimension of embeddings and hidden states |
| `N`          | 6     | Number of encoder/decoder layers          |
| `h`          | 8     | Number of attention heads                 |
| `d_ff`       | 2048  | Dimension of feed-forward network         |
| `seq_len`    | 350   | Maximum sequence length                   |
| `batch_size` | 8     | Training batch size                       |
| `num_epochs` | 20    | Number of training epochs                 |
| `lr`         | 1e-4  | Learning rate                             |
| `dropout`    | 0.1   | Dropout probability                       |

---

## 🚀 How It Works

### **Data Pipeline**

1. **Dataset**: Uses the OPUS Books parallel corpus (English-Italian)
2. **Tokenization**: Word-level tokenizers with special tokens `[UNK]`, `[PAD]`, `[SOS]`, `[EOS]`
3. **Masking**:
   - **Encoder mask**: Prevents attention to padding tokens
   - **Decoder mask**: Causal mask ensures the model only attends to previous positions during training

### **Training Process**

1. Loads bilingual sentence pairs from OPUS Books dataset
2. Tokenizes and pads sequences to fixed length (350 tokens)
3. Splits data into 90% training and 10% validation
4. Trains with Adam optimizer and cross-entropy loss (with label smoothing)
5. Saves checkpoints after each epoch
6. Logs training loss to TensorBoard

### **Translation Flow**

```
English sentence → Tokenize → Encoder →
                                        ↓
                                    Context vectors
                                        ↓
Italian tokens ← Decode ← Decoder ← Cross-attention
```

---

## 🔧 Installation & Usage

### **Requirements**

```bash
pip install torch torchvision
pip install datasets tokenizers
pip install tqdm tensorboard
```

### **Training the Model**

```bash
python train.py
```

The model will:

- Download the OPUS Books dataset automatically
- Build tokenizers if they don't exist
- Train for 20 epochs (configurable)
- Save checkpoints to `opus_books_weights/`
- Log metrics to TensorBoard

### **Monitoring Training**

```bash
tensorboard --logdir=runs
```

### **Resume Training**

The model automatically loads the latest checkpoint if `preload: "latest"` is set in config.

---

## 📊 Key Features

### **Attention Mechanism**

The multi-head attention allows the model to focus on different parts of the input sentence when generating each output word, capturing complex linguistic dependencies.

### **Residual Connections**

Each sub-layer (attention, feed-forward) uses residual connections to facilitate gradient flow:

```
output = LayerNorm(x + Sublayer(x))
```

### **Label Smoothing**

Improves generalization by preventing the model from becoming overconfident (ε = 0.1).

### **Checkpoint Management**

Saves model state, optimizer state, and training progress after each epoch for easy resumption.

---

## 🧠 What I Learned

Through this project, I gained hands-on understanding of:

1. **Self-Attention**: How models weigh the importance of different words in context
2. **Positional Encoding**: Encoding sequence order without recurrence
3. **Encoder-Decoder Architecture**: How source and target languages interact
4. **Masking Strategies**: Preventing information leakage during training
5. **Training Stability**: Importance of normalization, residual connections, and initialization
6. **Production ML**: Checkpoint management, logging, and reproducibility

---

## 📖 References

- **Paper**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- **Dataset**: OPUS Books Corpus (English-Italian parallel texts)
- **Framework**: PyTorch

---
 

## 🙏 Acknowledgments

This implementation is inspired by the original "Attention Is All You Need" paper and various educational resources on transformer architectures. The goal was to understand every component deeply by building it from the ground up.

---

**Built with 🧠 and ☕ to understand the magic behind modern LLMs!**
