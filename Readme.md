curl -X POST http://localhost:8005/generate_sql/ \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the total sales for 2020?"}'


pip install pandas datasets


docker build -t gino-llama-api .
docker run -p 8005:8005 gino-llama-api


