#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
LANGCHAIN_PROJECT="Rag_MultiQuery"


# In[2]:




# In[3]:


import fitz  # PyMuPDF
from langdetect import detect
import re

def clean_text(text):
    # Replace multiple whitespace with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    # Remove leading and trailing whitespace
    text = text.strip()
    
    return text

def extract_and_clean_english_text(pdf_path, max_page=66):
    """
    Extracts and cleans English text from a PDF up to a specified page.
    
    Args:
    pdf_path (str): The path to the PDF file.
    max_page (int): The maximum number of pages to process.

    Returns:
    str: The cleaned and concatenated English text extracted from the PDF.
    """
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)

    # Initialize a list to hold all English text
    english_text = []

    # Loop through each page in the PDF until the specified max_page
    for page_num in range(min(len(pdf_document), max_page)):
        page = pdf_document.load_page(page_num)
        
        # Extract text from the page
        text = page.get_text("text")
        
        # Split text into sentences/lines
        lines = re.split(r'\n', text)
        
        # Detect language and filter for English
        for line in lines:
            try:
                if detect(line) == 'en':
                    cleaned_line = clean_text(line)
                    if cleaned_line:  # Add only non-empty cleaned lines
                        english_text.append(cleaned_line)
            except:
                continue  # Skip lines where language detection fails

    return " ".join(english_text)

# Use the function
pdf_path = "washer.pdf"
english_text = extract_and_clean_english_text(pdf_path)
#print(english_text)


# In[4]:





# In[5]:


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma 
from langchain_openai import OpenAIEmbeddings



# Set embeddings
embd = OpenAIEmbeddings()

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=500
)

def split_text_into_chunks(text, chunk_size=2000, chunk_overlap=350):
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Split the text into chunks
    text_chunks = text_splitter.split_text(text)
    return text_chunks

# Use the functions
pdf_path = "washer.pdf"
cleaned_english_text = extract_and_clean_english_text(pdf_path)
text_chunks = split_text_into_chunks(cleaned_english_text)

# Print the first few chunks as a sample
for i, chunk in enumerate(text_chunks[::]):
    print(f"Chunk {i+1}:\n{chunk}\n")

# Add to vectorstore
vectorstore = Chroma.from_texts(
  texts=text_chunks,
  collection_name="rag-chroma",
  embedding=embd
)

retriever = vectorstore.as_retriever()


# In[6]:


# Verify the number of embeddings stored
  #print(f"Number of embeddings stored: {vectorstore.__len__()}")


# In[7]:


from langchain.prompts import ChatPromptTemplate

# Multi Query: Different Perspectives
template = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""
prompt_perspectives = ChatPromptTemplate.from_template(template)

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

generate_queries = (
    prompt_perspectives 
    | ChatOpenAI(temperature=0) 
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)


# In[11]:


from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

from langchain_core.prompts import ChatPromptTemplate


from langchain.load import dumps, loads

# Assuming generate_queries and retriever are already defined globally or within the function scope
# and are accessible to this cell

def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]

def answer_question(user_question):
    # Define the RAG template
    template = """You are a helpful assistant. Answer the question based only on the following context:
{context}

Answer the question based on the above context: {question}

Provide a detailed answer.
Do not justify your answers.
Do not give information not mentioned in the CONTEXT INFORMATION.
If you don't know the answer, say: "I can't answer this question since it is not mentioned in the context."""

    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_template(template)

    # Initialize ChatOpenAI for language generation
    llm = ChatOpenAI(temperature=0)

    # Retrieve documents using the retrieval_chain
    retrieval_chain = generate_queries | retriever.map() | get_unique_union
    docs = retrieval_chain.invoke({"question": user_question})

    # Build the final RAG chain
    final_rag_chain = (
        {"context": retrieval_chain,  # Pass retrieved documents as context
         "question": itemgetter("question")} 
        | prompt
        | llm
        | StrOutputParser()
    )

    # Invoke the final RAG chain to generate the response
    response = final_rag_chain.invoke({"question": user_question})

    return response



# In[ ]:




