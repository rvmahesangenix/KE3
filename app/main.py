from fastapi import FastAPI
from transformers import LlamaForSeq2SeqLM, LlamaTokenizer

app = FastAPI()
model_path = "/usr/share/ollama/.ollama/models/manifests/registry.ollama.ai/library/giniollama"
# Load the fine-tuned model
model = LlamaForSeq2SeqLM.from_pretrained('./results')  # Fine-tuned model path
tokenizer = LlamaTokenizer.from_pretrained(model_path)  # GinoLLama tokenizer

@app.post("/generate_sql/")
async def generate_sql(query: str):
    # Tokenize the input query
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    # Generate the SQL query
    output = model.generate(inputs['input_ids'], max_length=128)
    sql_query = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"sql_query": sql_query}

