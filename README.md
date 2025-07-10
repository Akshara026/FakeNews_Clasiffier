# FakeNews Clasiffier
wrk under progress

# Topic #
**Transformer Ensembles for Fake News Detection: A Multimodal Perspective with ViT, BERT, and DeBERTa
university**

- This project aims to accurately detect and classify both fake and real news content, as well as distinguish between AI-generated and authentic (real) images

# Model Details #
- In Our project we used  a hybrid ensemble deep learning models—BERT, DeBERTa, and Vision Transformer to perform multimodal fake news detection. Specifically, the system classifies textual news content as real or fake, and distinguishes between AI-generated images and authentic real-world images.

**BERT (Text Classification)**

We fine-tuned the BERT model (bert-base-uncased) on news articles. We combined the article title and content into one text input. The model learns to classify each article as REAL or FAKE.

-Tokenizer: BERT tokenizer with truncation and padding
-Loss Function: Weighted CrossEntropyLoss
-Optimizer: AdamW (lr = 2e-5)
-Evaluation Metrics: Accuracy, F1-score, Precision, Recall

**DeBERTa (Text Classification)**

DeBERTa is another transformer model we used to improve text classification results. It works similarly to BERT but gives better performance on longer and more complex text.

Tokenizer: DeBERTa tokenizer (max_length = 128)
Training Enhancements: Gradient checkpointing, FP32 precision
Evaluation Strategy: Validation every 500 steps using F1-score
Saving Strategy: Best model checkpoint saved based on F1

**ViT for Image Classification**

To detect whether an image is AI-generated or real, we fine-tuned the model using image data. The dataset is structured into two categories (real and fake) and preprocessed using ViTImageProcessor.

 Preprocessing: Pixel-level transformation using ViT’s image processor
 Training Strategy: Mixed-precision training (torch.amp) with early stopping
 Loss Function: CrossEntropyLoss
 Optimizer: AdamW with ReduceLROnPlateau scheduleru.



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
