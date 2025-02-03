from fastapi import FastAPI
from transformers import LlamaForSeq2SeqLM, LlamaTokenizer

app = FastAPI()

# Load the fine-tuned model
model = LlamaForSeq2SeqLM.from_pretrained('./results')  # Path to the fine-tuned model
tokenizer = LlamaTokenizer.from_pretrained("/usr/share/ollama/.ollama/models/manifests/registry.ollama.ai/library/giniollama")

@app.post("/generate_sql/")
async def generate_sql(query: str):
    # Tokenize the input query
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    # Generate the SQL query
    output = model.generate(inputs['input_ids'], max_length=128)
    sql_query = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"sql_query": sql_query}
