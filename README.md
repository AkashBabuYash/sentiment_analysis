# Sentiment Analysis of Amazon Reviews with BiLSTM

This project implements a **Bidirectional LSTM (BiLSTM)** deep learning model to classify Amazon product reviews as **Positive** or **Negative**. The model is trained on a subset of the Amazon Reviews dataset and achieves robust performance with text preprocessing, tokenization, and sequence padding.



## Overview

Sentiment analysis is a natural language processing (NLP) task that determines the emotional tone of a piece of text. This project uses a **Bidirectional LSTM** to capture context from both directions in a review, improving classification accuracy. The model is trained on 50,000 Amazon reviews and can be easily extended to larger datasets.

## Dataset

The dataset is obtained from Kaggle: [Amazon Reviews](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews) (downloaded via `kagglehub`). Each review contains:
- **label**: 1 (negative) or 2 (positive)
- **title**: review title
- **text**: full review text

Only the first 50,000 rows are used for faster experimentation. The labels are remapped to `0` (negative) and `1` (positive).

## Preprocessing

The following steps are applied to clean and prepare the text data:

1. **Combine title and text** into a single input string.
2. **Lowercase** all text.
3. **Remove HTML tags** (e.g., `<br />`).
4. **Remove non-alphabetic characters** (keep only a-z, A-Z).
5. **Remove English stopwords** using NLTK.
6. **Tokenization** with Keras Tokenizer (max vocabulary size = 10,000).
7. **Padding** sequences to a fixed length of 100 tokens.

## Model Architecture

The model is a sequential neural network built with TensorFlow/Keras:

| Layer                     | Output Shape       | Parameters |
|---------------------------|--------------------|------------|
| Embedding (10000 → 128)   | (None, 100, 128)   | 1,280,000  |
| Bidirectional LSTM (64)   | (None, 100, 128)   | 98,816     |
| Dropout (0.3)             | (None, 100, 128)   | 0          |
| Bidirectional LSTM (32)   | (None, 64)         | 41,216     |
| Dropout (0.3)             | (None, 64)         | 0          |
| Dense (1, sigmoid)        | (None, 1)          | 65         |

**Total parameters:** ~1.42 million (trainable)

- **Bidirectional LSTM**: Processes sequences forward and backward to capture richer context.
- **Dropout**: Regularization to prevent overfitting.
- **Sigmoid activation**: Binary classification output.

## Training

- **Loss function:** Binary crossentropy
- **Optimizer:** Adam
- **Batch size:** 64
- **Epochs:** 10 (with early stopping, patience=2)
- **Train/validation split:** 80/20

Early stopping monitors validation loss and restores the best weights.

## Evaluation

After training, the model outputs:
- Confusion matrix (actual vs. predicted labels)
- Accuracy, precision, recall (can be added via `sklearn.metrics`)

Example confusion matrix visualization is included in the notebook.

## Usage

### Requirements

Install the required libraries:

```bash
pip install kagglehub pandas tensorflow nltk scikit-learn matplotlib
