# ELI5 GPT-2 Language Model

This project implements a **decoder-only Transformer language model** trained on the **ELI5 dataset** using the GPT-2 tokenizer.

The goal of the project is to train a model that achieves **test perplexity lower than the baseline value of 390** provided in the course assignment.

This work was developed for the course:

**AI504 – Programming for AI**

---

# Project Overview

Language models estimate the probability of a sequence of tokens.

Given a sequence:

x₁, x₂, x₃, ..., xₙ

the model learns to predict the next token:

P(xₜ | x₁, x₂, ..., xₜ₋₁)

In this project we train a **custom GPT-2 style Transformer** on the **ELI5 (Explain Like I'm Five) dataset**, using only the **answer text** from the dataset.

The model is trained from scratch using PyTorch and the HuggingFace Transformers library.

---

# Dataset

The dataset is preprocessed using the provided `base.py` script.

Key characteristics:

* Source: **ELI5 (Explain Like I'm Five) dataset**
* Only the **answer field** is used
* Tokenizer: **GPT-2 tokenizer**
* Vocabulary size: **50,257 tokens**
* Fixed sequence length: **200 tokens**

Dataset sizes:

| Split      | Number of Samples |
| ---------- | ----------------- |
| Train      | 17,655            |
| Validation | 5,344             |
| Test       | 75                |

Important rule:

The **test set is never used during training or tuning**.

---

# Model Architecture

The model is a **decoder-only Transformer** similar to a small GPT-2 architecture.

Configuration:

* Embedding dimension: **512**
* Transformer layers: **6**
* Attention heads: **8**
* Context length: **200**
* Dropout: **0.1**

Each Transformer block contains:

1. Multi-Head Masked Self-Attention
2. Feed-Forward Network
3. Residual Connections
4. Layer Normalization

The model is implemented using **GPT2LMHeadModel** from the HuggingFace Transformers library.

---

# Training Setup

Training configuration:

Optimizer
AdamW

Learning rate
2e-4

Scheduler
Linear warmup + linear decay

Warmup steps
500

Batch size
32

Epochs
4

Gradient clipping
1.0

The model is trained using **cross-entropy loss for next-token prediction**.

---

# Results

The trained model was evaluated using the official evaluation script provided in the assignment.

Evaluation command:

```python
from test_lm import evaluate
evaluate("LLM_Model.npy")
```

Evaluation output:

```
Test dataset size: 75
20256306 - Perplexity: 210
```

### Performance Comparison

| Model                                   | Test Perplexity |
| --------------------------------------- | --------------- |
| Baseline model (provided in assignment) | 390             |
| Our GPT-2 based model                   | **210**         |

The trained model significantly improves over the baseline, reducing perplexity by **about 46%**.

Lower perplexity indicates that the model is better at predicting the next token in a sequence.

---

# Logits Generation

After training, the model generates **raw logits** for the test set.

Logits represent the model’s **unnormalized prediction scores** for each token in the vocabulary.

Output format:

```
(75, 200, 50257)
```

Where:

* 75 = number of test samples
* 200 = sequence length
* 50257 = GPT-2 vocabulary size

The logits are stored as:

```
float16
```

This reduces file size and allows the file to be uploaded to the course submission platform.

The file is saved as:

```
studentID.npy
```

---

# Repository Structure

```
.
├── base.py
├── studentID.py
├── studentID.ipynb
├── test_lm.py
└── README.md
```

File description:

**studentID.py**

Main script that:

* loads and preprocesses the dataset
* trains the Transformer language model
* generates logits for the test set
* saves the logits as a `.npy` file

**studentID.ipynb**

Google Colab notebook used to run the script in the required environment.

**test_lm.py**

Evaluation script used to compute the perplexity score from the generated logits.

---

# Running the Project

Clone the repository:

```
git clone https://github.com/yourusername/eli5-gpt2-language-model.git
cd eli5-gpt2-language-model
```

Run the training script:

```
python studentID.py
```

This will automatically:

1. preprocess the dataset
2. train the language model
3. generate logits for the test set
4. save the logits as:

```
studentID.npy
```

---

# Environment Requirements

The project requires the following versions:

Python
3.12.12

PyTorch
2.9.0

datasets
4.0.0

transformers
4.57.1

These versions ensure reproducibility within the **Google Colab Free environment**.

---

# Key Concepts Demonstrated

This project demonstrates several core NLP and deep learning concepts:

* Transformer language models
* Autoregressive text generation
* GPT-2 tokenizer usage
* PyTorch training pipelines
* Learning rate scheduling
* Gradient clipping
* Perplexity evaluation

---

# Author

Bushra Monika Hossain

Graduate School of AI
KAIST
