from transformers import BartForConditionalGeneration, BartTokenizer
from neo4j import GraphDatabase
from tabulate import tabulate
from datetime import datetime

def generate_query(model, tokenizer, input_text):
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

def format_value(value):
    if isinstance(value, bool):
        return "Yes" if value else "No"
    elif isinstance(value, (int, float)):
        if "salary" in str(value):
            return f"${value:,.2f}"
        elif "bonus" in str(value):
            return f"{value*100:.1f}%"
        return f"{value:,}"
    return str(value)

def beautify_results(results):
    if not results:
        return "No results found."
    
    # Get all unique properties from all records
    all_properties = set()
    for record in results:
        for value in record.values():
            # Check if it's a Neo4j Node
            if hasattr(value, 'labels') and hasattr(value, 'properties'):
                all_properties.update(value.properties.keys())
    
    # Convert results to table format
    table_data = []
    headers = results[0].keys()
    # Format headers (capitalize and replace underscores with spaces)
    formatted_headers = [h.replace('_', ' ').title() for h in headers]
    for record in results[1:]:
        table_data.append(record.values())
    
    
    if not table_data:
        return "No properties found in the results."
    
    # Generate table using tabulate
    table = tabulate(
        table_data,
        headers=formatted_headers,
        tablefmt='pretty',
        numalign='right',
        stralign='left'
    )
    
    summary = f"\nFound {len(table_data)} results"
    if len(results) != len(table_data):
        summary += f" (from {len(results)} records)"
    summary += ":\n"
    
    return summary + table

def execute_cypher_query(cypher_query):
    try:
        with driver.session() as session:
            results = session.run(cypher_query)
            records = [record["e"] for record in results]
            
            # Debug information
            if not records:
                print("\nPossible issues:")
                print("1. The query might be correct but no data matches the conditions")
                print("2. The query might need adjustments")
                print("\nTry these diagnostic queries:")
                print("1. To check if there's any data:")
                print("   MATCH (n) RETURN count(n)")
                print("2. To see sample data:")
                print("   MATCH (n) RETURN n LIMIT 5")
            
            return beautify_results(records)
    except Exception as e:
        print(f"\nError executing query: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Check if Neo4j is running")
        print("2. Verify the Cypher syntax")
        print("3. Ensure the node labels and relationship types exist")
        return f"Query execution failed: {str(e)}"

# Load fine-tuned model
try:
    model = BartForConditionalGeneration.from_pretrained("./neo4j_query_model/final")
    tokenizer = BartTokenizer.from_pretrained("./neo4j_query_model/final")
except Exception as e:
    print("Error loading fine-tuned model. Make sure training has completed successfully.")
    print(f"Error details: {str(e)}")

while True:
    try:
        query = input("\nEnter your query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
            
        result = generate_query(model, tokenizer, query)
        
        results = execute_cypher_query(result)
        print(results)
        
    except Exception as e:
        print(f"\nError: {str(e)}")

# Close Neo4j connection
driver.close()
