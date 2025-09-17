# rag_server.py

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

# --- Load the model and create the chains (as in your original code) ---
# NOTE: You'll need to define your embeddings and LLM here.
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models import ChatTongyi
from langchain_core.documents import Document

# Initialize LLM and Embeddings with your Qwen API key
# Get the API key from an environment variable for security
qwen_api_key = os.getenv("QWEN_API_KEY")

if not qwen_api_key:
    print("Error: QWEN_API_KEY environment variable not set.")
    exit()

os.environ["QWEN_API_KEY"] = qwen_api_key
embeddings = DashScopeEmbeddings(model="text-embedding-v2")
llm = ChatTongyi(model="qwen-turbo")

# Get your DashScope API key from Alibaba Cloud
if not os.environ.get("DASHSCOPE_API_KEY"):
    os.environ["DASHSCOPE_API_KEY"] = getpass.getpass("Enter API key for Alibaba Cloud DashScope: ")

# Initialize the embedding model
embeddings = DashScopeEmbeddings(model="text-embedding-v2")

# Initialize the Qwen LLM
llm = ChatTongyi(model="qwen-turbo")


# 1. Load documents from a text file
file_path = "nke-10k-2023.pdf"
if not os.path.exists(file_path):
    print(f"Error: PDF file not found at {file_path}")
    exit()

try:
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # 2. Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)

    # 3. Create a vector store using embeddings
    vector = FAISS.from_documents(documents, embeddings)

    # 4. Create a retriever from the vector store
    retriever = vector.as_retriever()

    # 5. Define the prompt template
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Use three sentences maximum and keep the answer concise.\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # 6. Create the chains
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

except Exception as e:
    print(f"An error occurred during chain creation: {e}")
    rag_chain = None

# --- Flask server setup ---
app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

@app.route('/ask', methods=['POST'])
def ask():
    if not rag_chain:
        return jsonify({"answer": "Error: RAG chain not initialized."}), 500
    
    data = request.json
    question = data.get('question')
    
    if not question:
        return jsonify({"answer": "Error: No question provided."}), 400

    print(f"Received question: {question}")
    
    try:
        response = rag_chain.invoke({"input": question})
        answer = response.get("answer", "No answer found.")
        return jsonify({"answer": answer})
    except Exception as e:
        print(f"Error invoking RAG chain: {e}")
        return jsonify({"answer": "An error occurred while processing your request."}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
