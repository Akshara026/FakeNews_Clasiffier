# Fake-news-Clasiffier
wrk under progress


Fake News Classifier (BERT-based)

This project detects whether a news article is real or fake using a fine-tuned BERT model for binary text classification.
Disclaimer

    This project is for educational and experimental purposes only.
    It is not suitable for real-world fact-checking or serious decision-making.
    The model uses a simple binary classifier and does not verify factual correctness.

Project Overview

This fake news classifier was built as part of a research internship to:

    Learn how to fine-tune transformer models on classification tasks
    Practice handling class imbalance using weighted loss
    Deploy models using Hugging Face-compatible APIs

How It Works

    A BERT-based model (bert-base-uncased) was fine-tuned on a labeled dataset of news articles.
    Input text is tokenized using BertTokenizer.
    A custom Trainer with class-weighted loss was used to handle class imbalance.
    Outputs are binary: 0 = FAKE, 1 = REAL.

Training Details

    Model: BertForSequenceClassification
    Epochs: 4
    Batch size: 8
    Learning rate: 2e-5
    Optimizer: AdamW (via Hugging Face Trainer)
    Evaluation Metrics: Accuracy, F1-score, Precision, Recall

🛠 Libraries Used

    transformers
    datasets
    torch
    scikit-learn
    pandas
    nltk (optional preprocessing)

📦 Installation & Running

pip install -r requirements.txt
python app.py

Or run the training script in a notebook or script environment if you're using Google Colab or Jupyter.
