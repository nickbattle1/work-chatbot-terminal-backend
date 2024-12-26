# Zorro: Axent's Workplace RAG AI Chatbot  

Zorro is a Retrieval-Augmented Generation (RAG) AI chatbot backend designed for Axent. It works out of the terminal as-is and can be connected to a custom frontend for full functionality. The prompt is tailored to Axent's specific needs but can be adapted for other businesses or applications.  

## Key Features and Architecture  

- Language Model: Utilizes Anthropic's Claude Haiku model for natural language processing and generation.  
- Embedding Model: Employs OpenAI's embedding model for creating high-dimensional vector representations of data.  
- Data Parsing and Formatting:  
  - Supports text-based data files such as JSON, CSV, Excel, Word, and PDF.  
  - Reformats files into a format compatible with the language model.  
- Data Chunking and Storage: Processes large datasets by chunking them and embedding the information into a vector database for efficient querying.  
- Customizable Prompt: The core prompt can be adapted to specific business needs, ensuring relevance across various applications.  

## Requirements  

- Anthropic API Key: Required for accessing the Claude Haiku language model.  
- OpenAI API Key: Required for generating embeddings.  
- Frontend Integration: A custom frontend is required to provide a user interface for interaction with the chatbot.  

## Usage Instructions  

1. Input your Anthropic API key and OpenAI API key into the Python code where indicated.  
2. Create a folder named `docs` within the project directory.  
3. Add your data files (e.g., JSON, CSV, Excel, Word, PDF) to the `docs` folder.  
4. Customize the prompt in the code to align with your business or application requirements.  
5. Run the Python script to initialize the backend.  
6. Connect the backend to a frontend interface to enable end-user interaction.  

This backend processes, embeds, and retrieves data efficiently, serving as the foundation for an intelligent chatbot system tailored to your business needs.  
