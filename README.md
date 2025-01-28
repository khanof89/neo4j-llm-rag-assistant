# 🚀 Neo4j LLM RAG Assistant

Transform natural language queries into Cypher using LLMs and build intelligent knowledge graph applications.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)]()
[![Neo4j](https://img.shields.io/badge/Neo4j-5.0%2B-blue)]()
[![Transformers](https://img.shields.io/badge/Transformers-latest-green)]()

## 🎯 Overview

This project demonstrates how to build an intelligent query system combining Large Language Models (LLM), Retrieval-Augmented Generation (RAG), and Knowledge Graphs using Neo4j. Perfect for developers and data scientists looking to build natural language interfaces for graph databases.

### 🌟 Key Features

- 🤖 Natural Language to Cypher query conversion using fine-tuned BART
- 📊 Automated data ingestion and categorization pipeline
- 🎯 Knowledge Graph creation and management
- 🔍 Intelligent query processing and retrieval
- 🎓 Educational examples using organizational data

## 🛠️ Components

1. `employees.csv`: Sample organizational dataset
2. `sync.py`: Data ingestion and Neo4j graph population script
3. `train-bart.py`: LLM training pipeline for natural language to Cypher conversion
4. `neo4j_client.py`: Neo4j interaction and query processing interface

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Configuration

1. Set up Neo4j database
2. Update connection settings in `neo4j_client.py`
3. Run data ingestion:
   ```bash
   python sync.py
   ```
4. Train the model:
   ```bash
   python train-bart.py
   ```

## 💡 Use Cases

- 🏢 Enterprise Knowledge Management
- 👥 HR Analytics and Organizational Insights
- 📊 Data Discovery and Exploration
- 🤖 Conversational AI Interfaces
- 🎯 Intelligent Search Systems

## 🔑 Keywords

`LLM` `RAG` `Neo4j` `Knowledge Graphs` `Natural Language Processing` `BART` `Fine-tuning` `Graph Database` `Cypher` `Transformers` `Enterprise Data` `HR Analytics` `Python` `Machine Learning` `AI` `Data Engineering` `Information Retrieval` `Semantic Search` `Data Science` `Graph AI`

## 📚 Learn More

This project serves as a practical example of:

- Building LLM-powered applications
- Implementing RAG systems
- Working with Knowledge Graphs
- Natural Language Understanding
- Enterprise Data Management

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=khanof89/neo4j-llm-rag-assistant&type=Date)](https://star-history.com/#khano8f9/neo4j-llm-rag-assistant&Date)

## 🔗 Related Projects

- [LangChain](https://github.com/hwchase17/langchain)
- [Neo4j Graph Data Science](https://github.com/neo4j/graph-data-science)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
