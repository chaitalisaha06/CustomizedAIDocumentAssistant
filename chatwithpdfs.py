from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import numpy as np
import PyPDF2
from redisvl.query import VectorQuery
import streamlit as st
from redisvl.index import SearchIndex
from redis import Redis
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import getpass
import os

os.environ["GROQ_API_KEY"] = "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"


st.title("CUSTOMIZED OWN CHATGPT")
# Center align the title using custom CSS
st.markdown(
    """
    <style>
    .centered-title {
        text-align: center;
        font-size: 3em;
        color: #4CAF50;
    }
    .centered-text {
        text-align: center;
        font-size: 1.2em;
    }
    .welcome-note {
        text-align: center;
        font-size: 1.5em;

        margin-top: 20px;
        margin-bottom: 20px;
    }
    .welcome-note1 {
        text-align: center;
        font-size: 0.7em;

        margin-top: 20px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display the welcome note
st.markdown('<div class="welcome-note">Welcome to Your Customized AI Document Assistant!</div>', unsafe_allow_html=True)
st.markdown('<div class="welcome-note1">Empower your productivity with our AI-driven Document Assistant. Seamlessly upload, manage, and interact with your documents like never before. Upload PDFs, extract valuable information, and engage in intuitive Q&A with your files. Experience the future of document management today!</div>', unsafe_allow_html=True)


schema = {
    "index": {
        "name": "vector_search",
        "prefix": "doc",
    },
    "fields": [
        {"name": "content", "type": "text"},
        {
            "name": "content_vector",
            "type": "vector",
            "attrs": {
                "dims": 384,
                "distance_metric": "cosine",
                "algorithm": "flat",
                "datatype": "float32",
            },
        },
    ],
}

index = SearchIndex.from_dict(schema)
index.connect("redis://default:8uWa4jpjKR7mIQYPF2qwQgEUWF0aYoO1@redis-15313.c301.ap-south-1-1.ec2.redns.redis-cloud.com:15313")

model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

option = st.selectbox("Choose your Action ", ["","Upload a PDF", "Chat from previous files"])

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page_number in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_number]
        text += page.extract_text()
    return text

# Upload PDF option
if option == "Upload a PDF":
    st.header("Upload a PDF")
    uploaded_file = st.file_uploader("UPLOAD DOCUMENTS HERE", accept_multiple_files=False, type=["txt", "pdf"])
    st.write("---")

    if st.button("Submit & Process") and uploaded_file is not None:
        with st.spinner("Processing..."):
            data = extract_text_from_pdf(uploaded_file)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
            chunks = text_splitter.split_text(data)

            index.create(overwrite=True)

            data = [
                {
                    'doc_id': f'{i}',
                    'content': chunk,
                    'content_vector': np.array([model.embed_query(chunk)], dtype=np.float32).tobytes(),
                }
                for i, chunk in enumerate(chunks)
            ]

            keys = index.load(data, id_field="doc_id")
            st.write("File uploaded")

# Chat with previous files option
elif option == "Chat from previous files":
    st.header("Chat from Uploaded Files")

    query = st.text_input("Enter your query related to uploaded files:")
    llm = ChatGroq(
            model="mixtral-8x7b-32768",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            # other params...
        )

    if query:
        query_embedding = model.embed_query(query)

        vector_query = VectorQuery(
            vector=query_embedding,
            vector_field_name="content_vector",
            num_results=5,
            return_fields=["doc_id", "content"],
            return_score=True,
        )

        result = index.query(vector_query)
        combined_content = "\n\n".join([item["content"] for item in result])
        # st.write(combined_content)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant, now give me the relevant answer from the context:  {context}.",
                ),
                ("human", "the question is {input}"),
            ]
        )

        chain = prompt | llm
        response=chain.invoke(
            {
                "context": combined_content,
                "input": query,

            }
        )
        st.write(response.content)
