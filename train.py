from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# 1. Load dataset
dataset = load_dataset("civil_comments")

# Reduce size (IMPORTANT for local system)
dataset["train"] = dataset["train"].select(range(10000))
dataset = dataset["train"].train_test_split(test_size=0.1)

# 2. Load tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 3. Preprocess
def preprocess(example):
    return tokenizer(example["text"], truncation=True, padding="max_length")

dataset = dataset.map(preprocess, batched=True)

# 4. Create labels
def format_labels(example):
    example["label"] = int(example["toxicity"] > 0.5)
    return example

dataset = dataset.map(format_labels)

# 5. Load model
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

# 7. Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }

# 8. Training config
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    num_train_epochs=1,
    logging_dir="./logs",
    learning_rate=2e-5
)

# 9. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics
)

# 10. Train
trainer.train()

# 11. Save model
model.save_pretrained("model")
tokenizer.save_pretrained("model")