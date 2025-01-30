import streamlit as st
from utils import (
    Neo4jConnection,
    load_llm_model,
    generate_query
)


st.title('Neo4j LLM RAG Assistant')

style = """
<style>
.stTextInput {
position: fixed;
bottom: 0;
width: 50%;
justify-content: center;
align-items: end;
margin-bottom: 0.5rem;
}

.stTable {
    height: 40rem;
    overflow-y: scroll
}
"""
st.markdown(style, unsafe_allow_html=True)

st.text_input('Enter your query', key='query')
 # Load the model
model, tokenizer = load_llm_model()

if not model or not tokenizer:
    print("Failed to load the model. Exiting...")

# Initialize Neo4j connection
neo4j = Neo4jConnection()

if st.session_state.query:
    # Generate Cypher query
    cypher_query = generate_query(model, tokenizer, st.session_state.query)
    print(f"\nGenerated Cypher: {cypher_query}")
                
    # Execute query and display results
    results = neo4j.execute_query(cypher_query, return_table=False)
    if results:
        st.table(results)
    