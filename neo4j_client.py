from transformers import BartForConditionalGeneration, BartTokenizer
from neo4j import GraphDatabase

def generate_query(model, tokenizer, input_text):
    print(input_text)
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(
        inputs['input_ids'],
        max_length=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=1,
        no_repeat_ngram_size=2
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Neo4j Connection
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "12345678"

# Set up Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

def execute_cypher_query(cypher_query):
    with driver.session() as session:
        results = session.run(cypher_query)
        return [record for record in results]
    
# Load fine-tuned model
try:
    model = BartForConditionalGeneration.from_pretrained("./neo4j_query_model/final")
    tokenizer = BartTokenizer.from_pretrained("./neo4j_query_model/final")
except Exception as e:
    print("Error loading fine-tuned model. Make sure training has completed successfully.")
    print(f"Error details: {str(e)}")

query = input("query")

result = generate_query(model, tokenizer, query)
print(result)
results = execute_cypher_query(query)
print("Results:", results)

# Close Neo4j connection
driver.close()
