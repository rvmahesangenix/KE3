# train.py

import pandas as pd
from datasets import Dataset
from transformers import LlamaForSeq2SeqLM, LlamaTokenizer, Trainer, TrainingArguments

# Load the CSV file
df = pd.read_csv("data/your_dataset.csv")  # Path to your CSV dataset
model_path = "/usr/share/ollama/.ollama/models/manifests/registry.ollama.ai/library/giniollama"
# Convert the CSV data to a Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Initialize the tokenizer and model for fine-tuning
model = LlamaForSeq2SeqLM.from_pretrained(model_path)  # Replace with the correct GinoLLama model path
tokenizer = LlamaTokenizer.from_pretrained(model_path)

# Tokenize the inputs and outputs
def tokenize_function(examples):
    inputs = tokenizer(examples['question'], padding='max_length', truncation=True)
    outputs = tokenizer(examples['sql_query'], padding='max_length', truncation=True)
    return {
        'input_ids': inputs['input_ids'],
        'labels': outputs['input_ids']
    }

# Apply the tokenizer to the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,  # Adjust for CPU usage
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets  # Or provide a separate validation dataset
)

# Train the model
trainer.train()
