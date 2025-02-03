import pandas as pd
from datasets import Dataset
from transformers import LlamaForSeq2SeqLM, LlamaTokenizer, Trainer, TrainingArguments

# Load the CSV file
df = pd.read_csv("your_dataset.csv")  # Replace with the path to your CSV

# Convert the CSV data to a Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Initialize the tokenizer and model for fine-tuning
model = LlamaForSeq2SeqLM.from_pretrained("/usr/share/ollama/.ollama/models/manifests/registry.ollama.ai/library/giniollama")  # Replace with the path to GinoLLama
tokenizer = LlamaTokenizer.from_pretrained("/usr/share/ollama/.ollama/models/manifests/registry.ollama.ai/library/giniollama")

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
    per_device_train_batch_size=4,  # Adjust batch size for CPU
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets  # You can define a separate validation set if available
)

# Train the model
trainer.train()
