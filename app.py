import torch                                    # PyTorch for GPU/CPU detection
import streamlit as st                          # Streamlit GUI framework
from dotenv import load_dotenv                  # Load environment variables from .env file
from PyPDF2 import PdfReader                    # Read text from PDF documents
from langchain.text_splitter import RecursiveCharacterTextSplitter        # Split large text into chunks
from langchain_community.vectorstores import FAISS               # Vector DB for text retrieval
from langchain_community.embeddings import HuggingFaceEmbeddings # Convert text chunks to embeddings
from langchain.memory import ConversationBufferMemory            # Store chat history
from langchain.chains import ConversationalRetrievalChain        # Conversational retrieval chain
from langchain_groq import ChatGroq                              # Groq LLM model
from langchain.chat_models import ChatOpenAI                     # OpenAI GPT models
from langchain.prompts import PromptTemplate                     # FIX: Needed for system prompt
from gtts import gTTS                                            # Google Text-to-Speech for voice output
import tempfile                                                  # Save temporary audio files for playback
import re                                                        # Regular expressions for text cleaning



# ===================== PDF PROCESSING FUNCTIONS =====================
# Functions to read PDF, split text, and create vector embeddings

def get_pdf_text(pdf_docs):
    """Extract text from all uploaded PDF files."""
    raw_text = ""
    for pdf in pdf_docs:
        pdf_doc = PdfReader(pdf)  
        for page in pdf_doc.pages:  
            page_text = page.extract_text() or ""
            page_text = re.sub(r"[^\S\r\n]+", " ", page_text)  # replace spaces/tabs but keep \n
            raw_text += page_text + " "
    return raw_text  

     
def get_chunks(text):
    """Split extracted text into manageable overlapping chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ".", " "],
    chunk_size=1600,
    chunk_overlap=250,
    length_function=len
    )

    return text_splitter.split_text(text)


def get_vector_store(text_chunks):
    """Convert text chunks to embeddings and store in FAISS vector store."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", 
        model_kwargs={"device": device}  # Pass device to model
    )
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)  # Build FAISS vector DB


# ===================== CONVERSATION CHAIN =====================
# Function to create the retrieval-based chat chain

def get_conversation_chain(vector_store):
    """Build a conversational retrieval chain using Groq LLM (or OpenAI)."""
    llm_1 = ChatOpenAI(model="gpt-4", temperature=0)  # OpenAI option
    # llm_1 = ChatGroq(model="llama-3.1-8b-instant", temperature=0)  # Groq option

    # FIX: Use PromptTemplate instead of plain string
    system_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are a helpful assistant for extracting information from a PDF.
        Answer the question using only the provided PDF content.
        Provide only the answer in a concise and summarized manner.
        If the answer is not present in the PDF, reply: 'I could not find the answer in the document.'
        
        Context: {context}
        Question: {question}
        """
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)  # Save chat history
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm_1,
        retriever=vector_store.as_retriever(
            search_type="mmr",      # Maximal Marginal Relevance (avoids redundancy)
            search_kwargs={"k": 5}  # return top 5 relevant chunks
        ),  # Connect vector store as retriever
        memory=memory,
        combine_docs_chain_kwargs={"prompt": system_prompt}  # FIX: now valid prompt
    )


# ===================== VOICE FUNCTIONS =====================
# Functions for text-to-speech in English and Sinhala

def translate_to_sinhala_with_gpt(answer: str) -> str:
    """Use OpenAI GPT model for Sinhala translation."""
    llm_2 = ChatOpenAI(model="gpt-4", temperature=0)
    response = llm_2.invoke(f"""Translate the following English text into Sinhala :.
                            \n\n{answer}""")
    
    # FIX: Simpler, safe handling
    return response.content.strip() if response else ""


def speak_english(answer):
    """English text + voice only."""
    if not answer.strip():  # FIX: Prevent gTTS crash on empty answer
        st.warning("No valid answer found for TTS.")
        return answer
    try:
        tts = gTTS(text=answer, lang="en", slow=False)   # Convert text to speech
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tts.save(tmp_file.name)                      # Save audio to temporary file
            st.audio(tmp_file.name, format="audio/mp3")  # Play audio in Streamlit
    except Exception as e:
        st.error(f"English TTS failed: {e}")
    
    st.markdown(f"**Answer:** {answer}")  # Display answer in text
    return answer


def speak_sinhala(answer):
    """Sinhala text + voice only (translated using GPT)."""
    sinhala_text = translate_to_sinhala_with_gpt(answer)  # Translate answer first

    if not sinhala_text.strip():  # FIX: Prevent gTTS crash on empty translation
        st.warning("No valid Sinhala answer found for TTS.")
        return sinhala_text

    try:
        # Shorten long Sinhala text (gTTS crashes on >5000 chars)
        if len(sinhala_text) > 4500:
            sinhala_text = sinhala_text[:4500]

        tts = gTTS(text=sinhala_text, lang="si", slow=False)  
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tts.save(tmp_file.name)
            st.audio(tmp_file.name, format="audio/mp3")
    except Exception as e:
        st.error(f"Sinhala TTS failed: {e}")

    st.markdown(f"**පිළිතුර:** {sinhala_text}")  # Display Sinhala answer
    return sinhala_text


# ===================== USER INPUT HANDLING =====================
# Function to manage user input, ask the model, and generate voice

def handle_user_input(user_question, voice_choice="English"):
    """Handle user question with conversational chain + voice output."""
    response = st.session_state.conversation({
        "question": user_question,
        "chat_history": st.session_state.chat_history or []
    })
    st.session_state.chat_history = response["chat_history"]  # Update chat history
    answer = response["answer"]

    # Voice + text response based on choice
    if voice_choice == "Sinhala":
        sinhala_text = speak_sinhala(answer)
        st.session_state.history_display.append(("User", user_question))
        st.session_state.history_display.append(("ChatBot (SI)", sinhala_text))
    else:
        english_text = speak_english(answer)
        st.session_state.history_display.append(("User", user_question))
        st.session_state.history_display.append(("ChatBot (EN)", english_text))

    return answer


# ===================== STREAMLIT APP =====================
# Main Streamlit UI code

def main():
    """Main Streamlit app."""
    load_dotenv()  
    st.set_page_config(page_title="Chat Bot PDF Reader", page_icon="📚", layout="wide")

    # Header section
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

    # Init session states to store conversation, chat history, and display
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "history_display" not in st.session_state:
        st.session_state.history_display = []

    # Sidebar + Main Layout columns
    sidebar_col, main_col = st.columns([1, 3])

    # ------------------ SIDEBAR ------------------
    with sidebar_col:
        st.subheader("📂 Upload PDFs")
        pdf_docs = st.file_uploader("Upload one or more PDF files", accept_multiple_files=True)
        if st.button("Process PDFs"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)          
                    text_chunks = get_chunks(raw_text)         
                    vector_store = get_vector_store(text_chunks)  
                    st.session_state.conversation = get_conversation_chain(vector_store)
                    st.session_state.chat_history = []     # FIX: reset chat history
                    st.session_state.history_display = []  # Reset history when new PDFs are uploaded
                    st.success("✅ PDFs processed successfully!")
            else:
                st.warning("Please upload at least one PDF file.")

        st.subheader("🔊 Voice Output")
        voice_choice = st.radio("Select language for voice output:", ["English", "Sinhala"], index=0)

    # ------------------ MAIN COLUMN ------------------
    with main_col:
        st.subheader("💬 Ask a Question")
        user_question = st.text_input("Type your question here and press Enter")
        if user_question and st.session_state.conversation:
            with st.spinner("Fetching response..."):
                handle_user_input(user_question, voice_choice)

        # Display chat history
        if st.session_state.history_display:
            st.subheader("📝 Chat History")
            for i, (role, msg) in enumerate(st.session_state.history_display):
                if role == "User":
                    st.markdown(f"<p style='color:#1f77b4'><b>{role}:</b> {msg}</p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<p style='color:#2ca02c'><b>{role}:</b> {msg}</p>", unsafe_allow_html=True)


# ===================== RUN APP =====================
if __name__ == "__main__":
    main()
