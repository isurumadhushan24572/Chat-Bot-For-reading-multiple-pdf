import torch                                    # PyTorch for GPU support
import streamlit as st                          # Develop the GUI
from dotenv import load_dotenv                  # Load environment variables
from PyPDF2 import PdfReader                    # Read PDF files
from langchain.text_splitter import CharacterTextSplitter        # Split the text into chunks
from langchain_community.vectorstores import FAISS               # Vector store for text chunks
from langchain_community.embeddings import HuggingFaceEmbeddings # Embeddings
from langchain.memory import ConversationBufferMemory            # Memory for chat
from langchain.chains import ConversationalRetrievalChain        # Conversational chain
from langchain.chat_models import ChatOpenAI                     # OpenAI GPT models
from langchain_groq import ChatGroq 


# Set page configuration
st.set_page_config(page_title="Chat Bot PDF Reader", page_icon="📚")


# Function to extract text from uploaded PDF files
def get_pdf_text(pdf_docs):
    raw_text = ""
    for pdf in pdf_docs:
        pdf_doc = PdfReader(pdf)
        for page in pdf_doc.pages:
            raw_text += page.extract_text()
    return raw_text          


# Function to split the extracted text into chunks
def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


# Function to create a vector store from text chunks
def get_vector_store(text_chunks):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # Use GPU if available
    embeddings = HuggingFaceEmbeddings(                                     # Use Hugging Face embeddings
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store


# Function to create a conversational chain
def get_conversation_chain(vector_store):

    # Intialize LLMS

    # llm = ChatOpenAI(model = "gpt-3.5-turbo", temperature = 0) 
    llm = ChatGroq(model="llama3-8b-8192",temperature = 0)
    # llm = ChatOpenAI(temperature=0.5)           # Define llm model
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),   # Convert vector_store as retriever
        memory=memory                            # give the chat history to the conversation_chain
    )
    return conversation_chain


# Function to handle user input
def handle_user_input(user_question):
    response = st.session_state.conversation({
        "question": user_question,
        "chat_history": st.session_state.chat_history or []  # Pass existing chat history
    })
    st.session_state.chat_history = response["chat_history"]  # Update chat history
    return response["answer"]


# Main Streamlit App Logic
def main():
    load_dotenv()                                # Load environment variables
    st.header("Chat Bot PDF Reader")

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []       # Initialize chat history

    # Sidebar for file upload and processing
    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)  # Allow multiple PDF files to be uploaded
        if st.button("Process Files"):
            with st.spinner("Processing..."):      # Show spinner while processing
                # Extract text from uploaded PDFs
                raw_text = get_pdf_text(pdf_docs)
                # st.write(raw_text)

                # Split text into chunks
                text_chunks = get_chunks(raw_text) 

                # Create vector store from text chunks
                vector_store = get_vector_store(text_chunks)

                # Create conversation chain and store it in session state
                st.session_state.conversation = get_conversation_chain(vector_store)
                st.success("Documents processed successfully!")

    # User question input and response display
    user_question = st.text_input("Enter your question here")  # User input
    if user_question and st.session_state.conversation:
        with st.spinner("Fetching response..."):
            response = handle_user_input(user_question)
            st.write(f"**Answer:** {response}")


# Run the app
if __name__ == "__main__":
    main()
