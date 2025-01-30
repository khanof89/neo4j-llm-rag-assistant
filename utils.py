from transformers import BartForConditionalGeneration, BartTokenizer
from neo4j import GraphDatabase
from tabulate import tabulate
from datetime import datetime

# Neo4j Connection Settings
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "12345678"

class Neo4jConnection:
    def __init__(self, uri=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self):
        self.driver.close()

    def execute_query(self, cypher_query, return_table=True):
        try:
            with self.driver.session() as session:
                results = session.run(cypher_query)
                records = [record["e"] for record in results]
                
                if not records:
                    print("\nPossible issues:")
                    print("1. The query might be correct but no data matches the conditions")
                    print("2. The query might need adjustments")
                    print("\nTry these diagnostic queries:")
                    print("1. To check if there's any data:")
                    print("   MATCH (n) RETURN count(n)")
                    print("2. To see sample data:")
                    print("   MATCH (n) RETURN n LIMIT 5")
                if return_table:
                    return beautify_results(records)
                else:
                    return records
        except Exception as e:
            print(f"\nError executing query: {str(e)}")
            print("\nTroubleshooting steps:")
            print("1. Check if Neo4j is running")
            print("2. Verify the Cypher syntax")
            print("3. Ensure the node labels and relationship types exist")
            return f"Query execution failed: {str(e)}"

def load_llm_model(model_path="./neo4j_query_model/final"):
    """Load the fine-tuned BART model and tokenizer."""
    try:
        model = BartForConditionalGeneration.from_pretrained(model_path)
        tokenizer = BartTokenizer.from_pretrained(model_path)
        return model, tokenizer
    except Exception as e:
        print("Error loading fine-tuned model. Make sure training has completed successfully.")
        print(f"Error details: {str(e)}")
        return None, None

def generate_query(model, tokenizer, input_text):
    """Generate Cypher query from natural language input."""
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

def format_value(value):
    """Format values for display."""
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
    """Format Neo4j results into a readable table."""
    if not results:
        return "No results found."
    
    # Get all unique properties from all records
    all_properties = set()
    for record in results:
        if hasattr(record, 'properties'):
            all_properties.update(record.properties.keys())
    
    # Convert results to table format
    table_data = []
    headers = sorted(list(all_properties))
    
    # Format headers (capitalize and replace underscores with spaces)
    formatted_headers = [h.replace('_', ' ').title() for h in headers]
    
    for record in results:
        if hasattr(record, 'properties'):
            row = []
            for header in headers:
                row.append(format_value(record.properties.get(header, '')))
            table_data.append(row)
    
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
    
    return f"\nFound {len(table_data)} results:\n{table}" 