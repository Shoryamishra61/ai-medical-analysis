import pandas as pd
import os
import gradio as gr
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import groq # Import the groq library to catch specific errors

# Load data from CSV
try:
    # Ensure medical_data.csv is in the same directory as app.py
    df = pd.read_csv('./medical_data.csv')
    print("Successfully loaded medical_data.csv")
except FileNotFoundError:
    print("Error: medical_data.csv not found. Make sure it's in the same directory as the script.")
    # Exit the script if the data file is not found, as it's essential
    exit()
except Exception as e:
    print(f"An error occurred while loading medical_data.csv: {e}")
    exit()


# Prepare context data from the first three columns
context_data = []
# Iterate through each row of the DataFrame
for i in range(len(df)):
  context = ""
  # Iterate through the first three columns (index 0, 1, 2)
  for j in range(3):
    # Check if the column index is valid for the DataFrame
    if j < len(df.columns):
      # Append column name
      context += df.columns[j]
      context += ": "
      # Access cell value by row and column index using .iloc
      # Convert value to string and handle potential missing values (NaN)
      cell_value = df.iloc[i, j]
      context += str(cell_value) if pd.notna(cell_value) else "N/A"
      context += " "
    else:
        # Break the inner loop if there are fewer than 3 columns
        break
  # Add the generated context string for the row to the list, removing trailing space
  context_data.append(context.strip())

print(f"Prepared context data for {len(context_data)} rows.")

# Get the Groq API key from the environment variable
# The langchain-groq library looks for 'GROQ_API_KEY' by default
groq_key = os.environ.get('GROQ_API_KEY')

# Initialize the LLM (Groq Chat Model)
llm = None # Initialize llm variable to None
try:
    # Check if the API key was found
    if not groq_key:
        # If key is not set, raise a ValueError with instructions
        raise ValueError("GROQ_API_KEY environment variable not set.")

    # Initialize the ChatGroq model
    # Trying 'llama3-70b-8192'. If this also fails, you MUST check the Groq console for available models.
    llm = ChatGroq(model="llama3-70b-8192", api_key=groq_key) # Changed model here
    print(f"Successfully initialized Groq model: llama3-70b-8192")

# Catch specific errors during LLM initialization
except ValueError as ve:
    print(f"Configuration Error: {ve}")
    print("Please set the GROQ_API_KEY environment variable correctly.")
    print("In Linux/macOS/Git Bash, use: export GROQ_API_KEY='your_groq_api_key'")
    print("In Windows Command Prompt, use: set GROQ_API_KEY=your_groq_api_key")
    exit() # Exit if API key is not set
except groq.BadRequestError as bre:
    print(f"Groq API Error: {bre}")
    # Check if the error is due to a decommissioned model or model not found
    if 'model_decommissioned' in str(bre) or 'model_not_found' in str(bre):
        print("The model specified is either decommissioned or does not exist/you do not have access.")
        print("Please update the 'model' parameter in ChatGroq to a currently supported and accessible model.")
        print("Check the Groq console or documentation for available models.")
    exit() # Exit on Groq API errors
except Exception as e:
    print(f"An unexpected error occurred during LLM initialization: {e}")
    exit() # Exit on other unexpected errors

# Initialize the Embedding model (HuggingFace)
embed_model = None # Initialize embed_model variable to None
try:
    # Using a robust embedding model from HuggingFace
    embed_model = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")
    print("Successfully loaded embedding model.")
except Exception as e:
    print(f"Error loading embedding model: {e}")
    print("Please ensure you have the 'transformers' and 'torch' (or 'tensorflow') libraries installed.")
    exit() # Exit if embedding model cannot be loaded

# Create or load the Chroma vector store
# This store will hold the embeddings of your medical data
vectorstore_dir = "./medical_dataset_store"
vectorstore = None # Initialize vectorstore variable to None

# Check if the vector store directory already exists
if not os.path.exists(vectorstore_dir):
    print("Creating new vector store...")
    try:
        # Create the vector store from the context data and embedding model
        vectorstore = Chroma.from_texts(
            texts=context_data,
            embedding=embed_model,
            collection_name="medical_dataset_store",
            persist_directory=vectorstore_dir, # Persist the store to disk
        )
        print("Vector store created and data added.")
    except Exception as e:
        print(f"Error creating vector store: {e}")
        exit() # Exit if vector store creation fails
else:
    # If the directory exists, load the existing vector store
    print("Loading existing vector store...")
    try:
        vectorstore = Chroma(
            collection_name="medical_dataset_store",
            embedding_function=embed_model,
            persist_directory=vectorstore_dir,
        )
        print("Vector store loaded.")
    except Exception as e:
        print(f"Error loading vector store: {e}")
        # If loading fails, it might be corrupted, attempt to recreate
        print("Attempting to recreate vector store due to loading error...")
        try:
             vectorstore = Chroma.from_texts(
                texts=context_data,
                embedding=embed_model,
                collection_name="medical_dataset_store",
                persist_directory=vectorstore_dir,
            )
             print("Vector store recreated.")
        except Exception as recreate_e:
             print(f"Failed to recreate vector store: {recreate_e}")
             exit() # Exit if recreation also fails


# Create a retriever from the vector store
# The retriever is used to fetch relevant documents based on a query
retriever = vectorstore.as_retriever()
print("Retriever created.")

# Define the RAG prompt template
# This template guides the LLM on how to use the retrieved context
template = ("""You are a medical expert.
    Use the provided context to answer the question.
    If you don't know the answer, say so. Explain your answer in detail.
    Do not discuss the context in your response; just provide the answer directly.

    Context: {context}

    Question: {question}

    Answer:""")

rag_prompt = PromptTemplate.from_template(template)
print("RAG prompt template defined.")

# Create the RAG chain using LangChain's Runnable interface
# This chain connects the retriever, prompt, and LLM
rag_chain = None # Initialize rag_chain variable to None
# Ensure LLM is successfully initialized before creating the chain
if llm is not None:
    rag_chain = (
        # Pass the retrieved context and the user's question to the prompt
        {"context": retriever, "question": RunnablePassthrough()}
        # Apply the RAG prompt
        | rag_prompt
        # Pass the result to the LLM
        | llm
        # Parse the LLM's output as a string
        | StrOutputParser()
    )
    print("RAG chain created.")
else:
    print("LLM was not initialized successfully. Cannot create RAG chain.")
    exit() # Exit if LLM initialization failed

# Gradio interface function for streaming the response
# This function is called when a user interacts with the Gradio app
def rag_memory_stream(text):
    partial_text = ""
    try:
        # Stream the response from the RAG chain
        for new_text in rag_chain.stream(text):
            partial_text += new_text
            # Yield the partial text to update the Gradio interface in real-time
            yield partial_text
    except Exception as e:
        # Handle errors during the streaming process
        yield f"An error occurred while streaming the response: {e}"
        print(f"Error during response streaming: {e}")


# Examples to display in the Gradio interface
examples = ['I feel dizzy', 'what is the possible sickness for fatigue']

# Define the Gradio interface
title = "Real-time AI App with Groq API and LangChain to Answer medical questions"
description = "Enter a medical question and get an answer based on the provided medical data."

demo = gr.Interface(
    title=title,
    description=description,
    fn=rag_memory_stream, # The function to call when the user submits input
    inputs="text", # The input component is a text box
    outputs="text", # The output component is a text box
    examples=examples, # Provide example inputs
    allow_flagging="never", # Disable the flagging feature
)

# Launch the Gradio app
if __name__ == "__main__":
    try:
        print("Launching Gradio app...")
        # Launch the web server for the Gradio app
        demo.launch()
    except Exception as e:
        print(f"An error occurred while launching the Gradio app: {e}")

