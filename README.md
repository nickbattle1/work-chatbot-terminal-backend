# Zorro: Axent's Workplace RAG AI Chatbot  

Zorro is a Retrieval-Augmented Generation (RAG) AI chatbot backend designed for Axent. It works out of the terminal as-is and can be connected to a custom frontend for full functionality. The prompt is tailored to Axent's specific needs but can be adapted for other businesses or applications.  

## Key Features and Architecture  

- **Language Model**: Utilizes Anthropic's Claude Haiku model for natural language processing and generation.  
- **Embedding Model**: Employs OpenAI's embedding model for creating high-dimensional vector representations of data.  
- **Data Parsing and Formatting**:  
  - Supports text-based data files such as JSON, CSV, Excel, Word, and PDF.  
  - Reformats files into a format compatible with the language model.  
- **Data Chunking and Storage**: Processes large datasets by chunking them and embedding the information into a vector database for efficient querying.  
- **Automatic File Hashing**: When documents are embedded into the database, the system automatically populates the `file_hashes.json` file to track the hashed values of the files for future reference.  
- **Customizable Prompt**: The core prompt can be adapted to specific business needs, ensuring relevance across various applications.  

## Requirements  

- **Anthropic API Key**: Required for accessing the Claude Haiku language model.  
- **OpenAI API Key**: Required for generating embeddings.  
- **OpenWeather API Key**: Required for weather queries.  

## Usage Instructions  

1. **Install Required Libraries**:  
   Install the necessary dependencies using:  
   ```bash  
   pip install -r requirements.txt  
   ```  

2. **Set Up API Keys**:  
   The project includes a `.env` file in the directory with placeholders for the API keys. Open the `.env` file and add your API keys as follows:  
   ```bash  
   OPENAI_API_KEY=your_openai_api_key  
   ANTHROPIC_API_KEY=your_anthropic_api_key  
   OPENWEATHER_API_KEY=your_openweather_api_key  
   ```  

3. **Prepare Data Files**:  
   - Create a folder named `docs` within the project directory.  
   - Add your data files (e.g., JSON, CSV, Excel, Word, PDF) to the `docs` folder.  

4. **Customize the Prompt**:  
   Modify the core prompt in the `app.py` file to align with your business or application requirements.  

5. **Run the Backend**:  
   Execute the Python script to initialize the backend system:  
   ```bash  
   python app.py  
   ```  

6. **Frontend Integration**:  
   Connect the backend to a frontend interface to enable end-user interaction.  

This backend processes, embeds, and retrieves data efficiently, serving as the foundation for an intelligent chatbot system tailored to your business needs.
