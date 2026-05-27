# Medical Chatbot

A Flask-based medical chatbot application that uses Retrieval Augmented Generation (RAG) to answer medical questions. The application leverages LangChain, Pinecone vector database, and Groq's LLM to provide accurate medical information.

## Features

- **RAG-based Architecture**: Uses Retrieval Augmented Generation for context-aware responses
- **PDF Document Processing**: Processes medical PDF documents and creates vector embeddings
- **Vector Search**: Uses Pinecone for efficient similarity search
- **Modern UI**: Clean and responsive chat interface
- **Real-time Chat**: Interactive chat interface with typing indicators

## Technology Stack

- **Backend**: Flask (Python web framework)
- **LLM**: Groq (Llama 3.1 8B Instant)
- **Vector Database**: Pinecone
- **Embeddings**: HuggingFace Sentence Transformers (all-MiniLM-L6-v2)
- **Document Processing**: LangChain
- **Frontend**: HTML, CSS, JavaScript, Bootstrap

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Pinecone API key
- Groq API key

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Medical-Chatbot
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory:
```env
PINECONE_API_KEY=your_pinecone_api_key
GROQ_API_KEY=your_groq_api_key
```

5. Place your medical PDF files in the `data/` directory

6. Create and populate the Pinecone index:
```bash
python store_index.py
```

7. Run the Flask application:
```bash
python app.py
```

8. Open your browser and navigate to `http://localhost:8080`

## Project Structure

```
Medical-Chatbot/
├── app.py                 # Main Flask application
├── setup.py              # Package setup configuration
├── .env                  # Environment variables
├── LICENSE               # Project license
├── README.md            # Project documentation (this file)
├── data/                # Directory for PDF documents
│   └── Medical_book.pdf
├── src/                 # Source code package
│   ├── __init__.py
│   ├── helper.py        # Helper functions for document processing
│   └── prompt.py        # Prompt templates
├── scripts/             # Utility and analysis scripts
│   ├── store_index.py   # Script to create and populate Pinecone index
│   ├── template.sh      # Template shell script
│   └── [analysis scripts]
├── research/            # Research and evaluation data
│   └── retrieval_evaluation_results.csv
├── docs/                # Documentation and assets
│   ├── *.md            # Documentation files
│   └── *.png           # Images and graphs
├── config/              # Configuration files
│   └── requirements.txt
├── templates/           # HTML templates
│   └── chat.html       # Chat interface template
├── static/             # Static files
│   └── style.css       # CSS styles
├── tests/              # Test files (future use)
└── medibot/            # Virtual environment
```

## Usage

1. Start the application by running `python app.py`
2. Open the web interface in your browser
3. Type your medical question in the chat input
4. Receive AI-powered responses based on the medical documents

## Configuration

- **Pinecone Index Name**: Can be changed in `app.py` (default: "medical-chatbot")
- **Chunk Size**: Adjustable in `src/helper.py` (default: 500 characters)
- **Retrieval Count**: Configurable in `app.py` (default: 3 documents)
- **LLM Model**: Changeable in `app.py` (default: "llama-3.1-8b-instant")

## Notes

- Ensure you have valid API keys for Pinecone and Groq
- The first run requires running `store_index.py` to create the vector index
- The application uses a 384-dimensional embedding model
- Responses are limited to 3 sentences for conciseness

## License

See LICENSE file for details.
