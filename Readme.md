curl -X POST http://localhost:8005/generate_sql/ \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the total sales for 2020?"}'


pip install pandas datasets



# Build the Docker image
docker build -t giniollama-text-to-sql .

# Run the container
docker run -p 8005:8005 giniollama-text-to-sql

