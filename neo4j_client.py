from utils import (
    Neo4jConnection,
    load_llm_model,
    generate_query
)

def main():
    # Load the model
    model, tokenizer = load_llm_model()
    if not model or not tokenizer:
        print("Failed to load the model. Exiting...")
        return

    # Initialize Neo4j connection
    neo4j = Neo4jConnection()

    try:
        while True:
            try:
                query = input("\nEnter your query (or 'exit' to quit): ")
                if query.lower() == 'exit':
                    break
                    
                # Generate Cypher query
                cypher_query = generate_query(model, tokenizer, query)
                print(f"\nGenerated Cypher: {cypher_query}")
                
                # Execute query and display results
                results = neo4j.execute_query(cypher_query)
                print(results)
                
            except Exception as e:
                print(f"\nError: {str(e)}")

    finally:
        neo4j.close()

if __name__ == "__main__":
    main()
