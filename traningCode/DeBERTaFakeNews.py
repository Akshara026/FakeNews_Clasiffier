import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch
from transformers import (
    DebertaTokenizer,
    DebertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# clears memory in gpu
torch.cuda.empty_cache()

# Loadin the dataset

df = pd.read_csv("\\home\\kaisex\\Desktop\\Deb\\Proper_Dataset.csv")
df['label'] = df['label'].str.upper().map({'FAKE': 0, 'REAL': 1})
df.dropna(subset=['text', 'label'], inplace=True)

# Splittin into train and test
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Tokenization with shorter sequences
tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")
def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=128,  # Reduced to 128 to prevent overflow
        padding=False
    )
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Loadin model with gradient checkpointing (FP32 precision)
model = DebertaForSequenceClassification.from_pretrained(
    "microsoft/deberta-base",
    num_labels=2,
    torch_dtype=torch.float32  # Explicitly use FP32 to prevent overflow
)
model.gradient_checkpointing_enable()

# Optimized training arguments (without FP16)
training_args = TrainingArguments(
    output_dir="./deberta_fake_news",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    logging_dir='./logs',
    logging_steps=100,
    fp16=False,  # Disabled FP16 to prevent overflow
    max_grad_norm=1.0,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none",
    optim="adamw_torch"  # Using standard AdamW instead of Adafactor
)

# Data collator with dynamic padding
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True,
    max_length=128,
    pad_to_multiple_of=8
)

# Metrics calculation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds)
    }

# Trainer with optimizations
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Startin the training
print("Starting training...")
trainer.train()
print("Training completed!")

# Evaluatin
print("\nEvaluating model...")
predictions = trainer.predict(test_dataset)
y_true = predictions.label_ids
y_pred = np.argmax(predictions.predictions, axis=1)
print(classification_report(y_true, y_pred, target_names=["FAKE", "REAL"]))

# Save model and tokenizer
save_path = "\\home\\kaisex\\Desktop\\Deb\\deberta_fake_news_model"
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)
print(f"Model saved to {save_path}") 


# we USED BELOW CODE TO GET THE RESULTS OF THE MODEL (WE RAN IT SEPARATELY AFTER TRAINING COZ OF TIME IT TOOK TO TRAIN THE MODEL)

# import torch
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from transformers import DebertaTokenizer, DebertaForSequenceClassification, Trainer
# from datasets import Dataset
# from sklearn.metrics import (
#     classification_report,
#     confusion_matrix,
#     ConfusionMatrixDisplay,
#     roc_curve,
#     auc
# )

# # Paths
# model_path = "deberta_fake_news_model"
# data_path = "C:\\Users\\student\\Downloads\\Proper_Dataset.csv"

# # Load model and tokenizer
# model = DebertaForSequenceClassification.from_pretrained(model_path)
# tokenizer = DebertaTokenizer.from_pretrained(model_path)

# # Load dataset and fix labels
# df = pd.read_csv(data_path)
# df['label'] = df['label'].str.upper().map({'FAKE': 0, 'REAL': 1})
# df.dropna(subset=['text', 'label'], inplace=True)

# # Use 20% as test set
# from sklearn.model_selection import train_test_split
# _, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# # Create Hugging Face Dataset
# test_dataset = Dataset.from_pandas(test_df)

# # Tokenization
# def tokenize_function(example):
#     return tokenizer(
#         example["text"],
#         truncation=True,
#         max_length=128,
#         padding="max_length"
#     )

# test_dataset = test_dataset.map(tokenize_function, batched=True)

# # Set format for PyTorch
# test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# # Inference using Trainer
# trainer = Trainer(model=model)
# predictions = trainer.predict(test_dataset)

# # Predictions
# y_true = predictions.label_ids
# y_pred = np.argmax(predictions.predictions, axis=1)
# y_probs = predictions.predictions[:, 1]

# # Ensure no None
# if y_true is None or y_pred is None:
#     raise ValueError("Prediction failed: y_true or y_pred is None.")

# # Classification Report
# print("\nClassification Report:\n")
# print(classification_report(y_true, y_pred, target_names=["FAKE", "REAL"]))

# # Confusion Matrix
# cm = confusion_matrix(y_true, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["FAKE", "REAL"])
# disp.plot(cmap=plt.cm.Purples)
# plt.title("Confusion Matrix")
# plt.savefig("confusion_matrix.png")
# plt.show()

# # ROC Curve
# fpr, tpr, _ = roc_curve(y_true, y_probs)
# roc_auc = auc(fpr, tpr)

# plt.figure()
# plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
# plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve")
# plt.legend(loc="lower right")
# plt.savefig("roc_curve.png")
# plt.show()

