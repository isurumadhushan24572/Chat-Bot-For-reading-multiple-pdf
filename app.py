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
from langchain.chat_models import ChatOpenAI                     # OpenAI GPT models
from gtts import gTTS                                            # Google Text-to-Speech for voice output
import tempfile                                                  # Save temporary audio files for playback
from langchain.prompts import ChatPromptTemplate                 # Custom chat prompts


# ===================== PDF PROCESSING FUNCTIONS =====================

def get_pdf_text(pdf_docs):
    """Extract text from all uploaded PDF files."""
    raw_text = ""
    for pdf in pdf_docs:
        pdf_doc = PdfReader(pdf)  
        for page in pdf_doc.pages:  
            raw_text += page.extract_text() or ""  
    return raw_text          


def get_chunks(text):
    """Split extracted text into manageable overlapping chunks."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    return text_splitter.split_text(text)


def get_vector_store(text_chunks):
    """Convert text chunks to embeddings and store in FAISS vector store."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)


# ===================== CONVERSATION CHAIN =====================


def get_conversation_chain(vector_store):
    """Build a conversational retrieval chain using Groq LLM (or OpenAI)."""
    llm_1 = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)  # OpenAI option
    # llm = ChatGroq(model="llama-3.1-8b-instant",temperature = 0)    # Define llm model

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm_1,
        retriever=vector_store.as_retriever(),
        memory=memory
    )

# ===================== VOICE FUNCTIONS =====================

def translate_to_sinhala_with_gpt(answer: str) -> str:
    """Use OpenAI GPT model for Sinhala translation."""
    llm_2 = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm_2.invoke(f"""Translate the following English text into natural Sinhala :
                            Also, make sure the translation is suitable for text-to-speech synthesis.
                            And avoid transliteration; use proper Sinhala script.
                            And provide only answer without any additional information with a more summarized version.
                            \n\n{answer}""")
    
    # Extract response content
    if hasattr(response, "content"):
        return response.content.strip()
    elif isinstance(response, str):
        return response.strip()
    return str(response)


def speak_english(answer):
    """English text + voice only."""
    try:
        tts = gTTS(text=answer, lang="en", slow=False)   
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tts.save(tmp_file.name)
            st.audio(tmp_file.name, format="audio/mp3")
    except Exception as e:
        st.error(f"English TTS failed: {e}")
    
    st.markdown(f"**Answer:** {answer}")
    return answer


def speak_sinhala(answer):
    """Sinhala text + voice only (translated using GPT)."""
    sinhala_text = translate_to_sinhala_with_gpt(answer)

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

    st.markdown(f"**පිළිතුර:** {sinhala_text}")
    return sinhala_text


# ===================== USER INPUT HANDLING =====================

def handle_user_input(user_question, voice_choice="English"):
    """Handle user question with conversational chain + voice output."""
    response = st.session_state.conversation({
        "question": user_question,
        "chat_history": st.session_state.chat_history or []
    })
    st.session_state.chat_history = response["chat_history"]
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

def main():
    """Main Streamlit app."""
    load_dotenv()  
    st.set_page_config(page_title="Chat Bot PDF Reader", page_icon="📚", layout="wide")

    # Header
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

    # Init session states
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "history_display" not in st.session_state:   # For nice chat history display
        st.session_state.history_display = []

    # Sidebar + Main Layout
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
                    st.session_state.history_display = []   # reset history on new PDFs
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
