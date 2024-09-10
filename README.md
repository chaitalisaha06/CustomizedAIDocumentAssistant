# CustomizedAIDocumentAssistant
This project allows you to upload, manage, and interact with your documents using a customized AI-based assistant. You can upload PDFs, extract text, and interact with them using natural language queries.

## Prerequisites
To run this project, ensure you have the following dependencies installed:

1. Python (version 3.7+)
2. Packages (Install using pip install -r requirements.txt):
3. Streamlit
4. PyPDF2
5. SentenceTransformer
6. HuggingFaceEmbeddings (langchain_community.embeddings)
7. redis-py
8. redisvl
9. langchain (core, community)
10. groq (langchain_groq)

    
## Setup Instructions
### Clone the repository:

git clone <repository-url>
cd <repository-directory>




### Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate




### Install the required dependencies:


pip install -r requirements.txt




### Set up the environment variable for the Groq API key. You can either set it directly in your terminal:

export GROQ_API_KEY="[GROQ_API_KEY]"




### Or you can add it in the script as shown in the project file:

os.environ["GROQ_API_KEY"] = "[GROQ_API_KEY]"




### Set up Redis as your vector database. The project connects to a remote Redis instance (you can set up your own or use a managed Redis cloud). Ensure you have the connection details:

index.connect("redis://<'username'>:<'password'>@<'host'>:<'port'>")




### Running the Application
Run the Streamlit application:

streamlit run app.py




## Features
Upload a PDF: You can upload a PDF file, and the text will be extracted and stored in the Redis vector database for future interaction.

Chat with Uploaded Files: Use natural language queries to interact with your uploaded documents. The system retrieves relevant document sections using vector search and generates a response based on the context.


## Usage
Upload a PDF: Select "Upload a PDF" and choose the document you want to upload. Click on the Submit & Process button to extract and store the text.

Chat with Files: Enter a query to interact with the uploaded files. The AI will retrieve relevant sections and respond to your query.
