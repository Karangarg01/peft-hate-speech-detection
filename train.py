from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.nn import CrossEntropyLoss

# 1. Load dataset
dataset = load_dataset("civil_comments")

dataset["train"] = dataset["train"].select(range(50000))

dataset = dataset["train"].train_test_split(test_size=0.3)

# Load tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Preprocess
def preprocess(example):
    return tokenizer(example["text"], truncation=True, padding="max_length")

dataset = dataset.map(preprocess, batched=True)

# Create labels
def format_labels(example):
    example["label"] = int(example["toxicity"] > 0.5)
    return example

dataset = dataset.map(format_labels)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 6. Apply LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_lin", "v_lin"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS"
)

model = get_peft_model(model, lora_config)

# Class weights (handle imbalance)
import torch
from torch.nn import CrossEntropyLoss

class_weights = torch.tensor([1.0, 5.0])

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = CrossEntropyLoss(weight=class_weights.to(logits.device))
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    logging_dir="./logs",
    learning_rate=2e-5,
    fp16=torch.cuda.is_available(),
    dataloader_pin_memory=False
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics
)

trainer.train()

model.save_pretrained("model")
tokenizer.save_pretrained("model")