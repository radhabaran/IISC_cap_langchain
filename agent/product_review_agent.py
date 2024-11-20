# ***********************************************************************************************
# Instruction for using the program
# ***********************************************************************************************
# Please make sure the embeddings.npy file is available in data folder
# Please make sure the documents.pkl file is available in data folder
# Please set the path appropriately inside the program. You will find the below two statements 
# where you need to mention the correct path name.
# embedding_path = '/workspaces/IISC_cap_langchain/data/embeddings.npy'
# documents_path = '/workspaces/IISC_cap_langchain/documents.pkl'
# ***********************************************************************************************

import openai
import numpy as np
import pandas as pd
from openai import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough

import faiss
import warnings
import os

warnings.filterwarnings("ignore")
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
llm = None
memory = None
# vectorstore = None

def initialize_product_review_agent(llm_instance, memory_instance):
    global llm, memory
    llm = llm_instance
    memory = memory_instance
    logger.info("product_review agent initialized successfully")


def process(query):

    # Initialize the OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    System_Prompt = """
    Role
    You are a knowledgeable and compassionate customer support chatbot specializing various
    products in Amazon. Your goal is to provide accurate, concise, and empathetic information about
    various produts available in Amazon product catlogue and common delivery issues.
    Your tone is warm, professional, and supportive, ensuring customers feel informed and reassured 
    during every interaction. 

    Instructions
    Product Availability: For inquiries about product availability ,retrieve the data and 
    offer simple, friendly explanations. Use relatable terms, and analogies if needed, to 
    help customers understand the process.
    Cost Calculations: Guide customers in providing cost of the product based on the provided 
    information and explain any factors influencing the cost.
    Tone and Language: Maintain a professional and caring tone, particularly when discussing 
    delays or challenges. Show understanding and reassurance.

    Context
    You are the primary customer service chatbot in Amazon specializing on various products and
    services available in Amazon. You handle interactions with customers and retail businesses that often 
    have urgent concerns about the quality and timely delivery of their product shipments. 
    Providing accurate and clear updates, coupled with empathy , is crucial for building 
    trust and confidence.

    Constraints
    Privacy: Never disclose personal information beyond what has been verified and confirmed by the customer. Always ask for consent before discussing details
    about shipments.
    Conciseness: Ensure responses are clear and concise, avoiding jargon unless necessary for 
    context.
    Empathy in Communication: Always start the conversation politely and with empathy. 
    When addressing delays or challenges, prioritize empathy and acknowledge the customer's concern.
    Provide next steps and resasssurance. Always end the communication with Thank you for reaching 
    out to us and we value your doing business with us. 
    Accuracy: Ensure all product updates, cost estimates, and shipment details are
    accurate and up-to-date. If you do not have data on any of the details asked, 
    politely say I do not know. If the query is outside electronics and home care products, 
    politely and clearly say I do not know.
    Jargon-Free Language: Use simple language to explain logistics terms or processes if asked 
    to customers, particularly when dealing with high cost and fragile products.

    Examples
    Product price Inquiry

    Customer: "How much will it cost to purchase a product?"
    Your Response: "I'd be glad to calculate that for you! Could you also share the
    product name and model? Once I have those details, I'll provide an estimate and
    explain any other options."

    Issue Resolution for Delayed product Shipment

    Customer: "I am worried about the  delayed Amazon shipment."
    Your Response: "I undersatnd your concern, and I'm here to help. Let me check the
    status of your shipment. If needed, we'll coordinate with the carrier to ensure
    your product's safety and provide you with updates along the way."

    Proactive Update Offer

    Customer: "Can I get updates on my product shipment's address."
    Your Response: "Absolutely! I can send you notification whenever your product's shipment
    reaches a checkpoint or if there are any major updates. Would you like to set that
    up ?"

    Out of context queries

    Customer: "Could you please help me locate a doctor who can help diagnose my disease"
    Your Response: "I understand you are looking for a doctor. I just would like to inform you
    we do not provide any such services in Amazon. Request to find a good doctore and get the
    diagnosis done. We wish you recover faster and stay healthy. In case you have any question
    on Amazon products and services, I am more than happy to assist you.
    """

    # Get existing chat history from memory
    chat_history = ""
    if memory and memory.chat_memory.messages:
        chat_history = "\nPrevious conversation:\n"
        for msg in memory.chat_memory.messages:
            chat_history += f"{msg.content}\n"

    # Check if embeddings already exist
    embedding_path = '/workspaces/IISC_cap_langchain/data/embeddings.npy'
    documents_path = '/workspaces/IISC_cap_langchain/documents.pkl'

    # Modify the get_embedding function to use LangChain's OpenAIEmbeddings
    def get_embedding(text, engine="text-embedding-ada-002"):
        return embeddings.embed_query(text)

    try:
        if not os.path.exists(embedding_path):
            raise FileNotFoundError(f"Embedding file not found at: {embedding_path}")
        if not os.path.exists(documents_path):
            raise FileNotFoundError(f"Documents file not found at: {documents_path}")
    except FileNotFoundError as e:
        logger.error(str(e))
        raise

    if os.path.exists(embedding_path) and os.path.exists(documents_path):
        # Load existing embeddings and documents
        embeddings_list = np.load(embedding_path)
        with open(documents_path, 'rb') as f:
            documents = pickle.load(f)


    # Create FAISS index with faster search
    embeddings_np = np.array(embeddings_list).astype('float32')
    index=faiss.IndexFlatL2(len(embeddings_list[0]))
    index.add(embeddings_np)

    query_embedding = get_embedding(query)
    query_embedding_np = np.array([query_embedding]).astype('float32')

    _, indices = index.search(query_embedding_np, 2)
    retrieved_docs = [documents[i] for i in indices[0]]
    context = ' '.join(retrieved_docs)
    print("context retrieved :", context)
    print('*' * 100)

    # Include chat history in the prompt for context
    structured_prompt = f"""
    Context:
    {context}

    {chat_history}
    
    Current Query:
    {query}
    """

    print("structured prompt created :", structured_prompt)
    print('*' * 100)
    # Create messages for the chat model
    messages = [
        {"role": "system", "content": System_Prompt},
        {"role": "user", "content": structured_prompt}
    ]

    # For chat completion, you can use LangChain's ChatOpenAI
    chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
    response = chat_model.invoke(messages).content

    # Update memory with the current interaction
    if memory:
        memory.save_context({"input": query}, {"output": response})

    return response


def clear_context():
    memory.clear()