
import torch                                    # PyTorch for GPU/CPU detection
import streamlit as st                          # Streamlit GUI framework
from dotenv import load_dotenv                  # Load environment variables from .env file
from PyPDF2 import PdfReader                    # Read text from PDF documents
from langchain.text_splitter import CharacterTextSplitter        # Split large text into chunks
from langchain_community.vectorstores import FAISS               # Vector DB for text retrieval
from langchain_community.embeddings import HuggingFaceEmbeddings # Convert text chunks to embeddings
from langchain.memory import ConversationBufferMemory            # Store chat history
from langchain.chains import ConversationalRetrievalChain        # Conversational retrieval chain
from langchain_groq import ChatGroq                              # Groq LLM model
from gtts import gTTS                                            # Google Text-to-Speech for voice output
from googletrans import Translator                               # Translate English → Sinhala
import tempfile                                                  # Save temporary audio files for playback


# ===================== PDF PROCESSING FUNCTIONS =====================

def get_pdf_text(pdf_docs):
    """
    Extract text from all uploaded PDF files.
    Input: List of uploaded PDF files (streamlit file uploader)
    Output: Concatenated raw text string
    """
    raw_text = ""
    for pdf in pdf_docs:
        pdf_doc = PdfReader(pdf)  # Load PDF file
        for page in pdf_doc.pages:  # Iterate pages
            raw_text += page.extract_text()  # Extract text per page
    return raw_text          


def get_chunks(text):
    """
    Split extracted text into manageable overlapping chunks.
    Input: Raw text string
    Output: List of text chunks
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",       # Prefer splitting at newlines
        chunk_size=1000,      # Max 1000 characters per chunk
        chunk_overlap=100,    # Overlap to preserve context
        length_function=len   # Use Python len() function
    )
    return text_splitter.split_text(text)


def get_vector_store(text_chunks):
    """
    Convert text chunks to embeddings and store in FAISS vector store.
    Input: List of text chunks
    Output: FAISS vector store for retrieval
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU if available
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)


def get_conversation_chain(vector_store):
    """
    Build a conversational retrieval chain using Groq LLM.
    Input: FAISS vector store
    Output: ConversationalRetrievalChain object
    """
    # Intialize LLMS 
    # llm = ChatOpenAI(model = "gpt-3.5-turbo", temperature = 0) 
    # Define llm model # llm = ChatOpenAI(temperature=0.5) # Define llm model
    llm = ChatGroq(model="llama3-8b-8192",temperature = 0) 

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )


# ===================== VOICE FUNCTIONS =====================

def speak_english(answer):
    """
    Convert English answer to spoken voice.
    Input: String (English answer)
    Output: Plays audio in Streamlit
    """
    tts = gTTS(text=answer, lang="en", slow=True)   # Slow for clarity
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tts.save(tmp_file.name)
        st.audio(tmp_file.name, format="audio/mp3")


def speak_sinhala(answer):
    """
    Translate English answer to Sinhala and play Sinhala voice.
    Input: String (English answer)
    Output: Plays translated audio in Streamlit
    """
    translator = Translator()
    translated = translator.translate(answer, src="en", dest="si")  # Translate EN → SI
    sinhala_text = translated.text

    tts = gTTS(text=sinhala_text, lang="si", slow=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tts.save(tmp_file.name)
        st.audio(tmp_file.name, format="audio/mp3")

    return sinhala_text  # Return translated text (optional display)


# ===================== USER INPUT HANDLING =====================

def handle_user_input(user_question, voice_choice="English"):
    """
    Handle user question:
    - Pass question to conversational chain
    - Retrieve answer
    - Play audio in selected language
    Inputs:
        user_question: str
        voice_choice: "English" or "Sinhala"
    Output:
        Returns answer text and plays audio
    """
    response = st.session_state.conversation({
        "question": user_question,
        "chat_history": st.session_state.chat_history or []
    })
    st.session_state.chat_history = response["chat_history"]
    answer = response["answer"]

    # Play voice and display text
    if voice_choice == "Sinhala":
        sinhala_text = speak_sinhala(answer)
        st.markdown(f"**Answer (English):** {answer}")
        st.markdown(f"**Answer (Sinhala Translation):** {sinhala_text}")
    else:
        speak_english(answer)
        st.markdown(f"**Answer:** {answer}")

    return answer


# ===================== STREAMLIT APP =====================

def main():
    """
    Main Streamlit app function
    - Layout: Wide with sidebar for PDFs & voice selection
    - Main column for Q&A
    """
    load_dotenv()  # Load .env variables
    st.set_page_config(page_title="Chat Bot PDF Reader", page_icon="📚", layout="wide")

    # Header with styling
    st.markdown(
        """
        <h1 style='text-align: center; color: #4B8BBE;'>
        📚 PDF Chat Bot with Voice
        </h1>
        <p style='text-align: center; color: #306998;'>
        Ask questions from your uploaded PDFs and hear answers in English or Sinhala
        </p>
        """,
        unsafe_allow_html=True
    )

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Layout: Sidebar + Main Column
    sidebar_col, main_col = st.columns([1, 3])

    # ------------------ SIDEBAR ------------------
    with sidebar_col:
        st.subheader("📂 Upload PDFs")
        pdf_docs = st.file_uploader("Upload one or more PDF files", accept_multiple_files=True)
        if st.button("Process PDFs"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)          # Extract text
                    text_chunks = get_chunks(raw_text)         # Split into chunks
                    vector_store = get_vector_store(text_chunks)  # Build embeddings + FAISS
                    st.session_state.conversation = get_conversation_chain(vector_store)
                    st.success("✅ PDFs processed successfully!")
            else:
                st.warning("Please upload at least one PDF file.")

        # Voice selection radio button
        st.subheader("🔊 Voice Output")
        voice_choice = st.radio("Select language for voice output:", ["English", "Sinhala"], index=0)

    # ------------------ MAIN COLUMN ------------------
    with main_col:
        st.subheader("💬 Ask a Question")
        user_question = st.text_input("Type your question here and press Enter")
        if user_question and st.session_state.conversation:
            with st.spinner("Fetching response..."):
                handle_user_input(user_question, voice_choice)

        # Optional: Show expandable chat history
        if st.session_state.chat_history:
            with st.expander("📝 Chat History"):
                for i, msg in enumerate(st.session_state.chat_history):
                    st.markdown(f"{i+1}. **{msg['role']}**: {msg['content']}")


# ===================== RUN APP =====================
if __name__ == "__main__":
    main()
