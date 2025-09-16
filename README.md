# PDF Chatbot 💬📄

An intelligent chatbot that extracts instructions from multiple PDF files and provides accurate answers in real-time both Sinhala and English Languages using both text and voice.  

## 🚀 Features  
- **Multi-PDF Support**: Handles multiple PDFs to provide comprehensive responses.  
- **GPT-Powered**: Uses OpenAI's `gpt-4` model for natural and context-aware interactions.  
- **Embeddings with HuggingFace**: Incorporates `sentence-transformers/all-MiniLM-L6-v2` for robust text embedding.  
- **Chat History**: Conversation history managed seamlessly using `ConversationBufferMemory`.  
- **Vector Store**: Efficiently retrieves data using `FAISS`.  
- **Web Interface**: Simple and user-friendly UI built with `Streamlit`.  

---

## 🛠️ Technologies Used  

- **LLM**: [OpenAI GPT-4, GPT-4o-mini](https://openai.com/)  
- **Embeddings**: [HuggingFace Embeddings](https://huggingface.co/models)  
- **Framework**: [LangChain](https://langchain.readthedocs.io/)  
- **Vector Database**: [FAISS](https://github.com/facebookresearch/faiss)  
- **Web Development**: [Streamlit](https://streamlit.io/)  

---

## 🎯 How It Works  
1. Upload your PDF files.  
2. Ask questions or get instructions.  
3. The chatbot fetches relevant context from the PDFs and provides precise answers.  

---

## Steps Should Follow

create Virtual env

  ```
  python -m venv venv
  ```
activate Virtual env

  ```
  cd venv\Scripts\activate
  ```
creating .gitignore file , .env file and chatbot.py

  ```
  echo "# Files to ignore" > .gitignore && echo "OPENAI_API_KEY=your_api_key_here" > .env && echo "# Chatbot script" > app.py
  ```
Insatalling Required dependencies

  ```
  pip install -r requirements.txt
  ```

Save OpenAI API Key inside .env file
  ```
  OPENAI_API_KEY=your_openai_api_key
  ```

Run Chatbot.py file

  ```
  streamlit run app.py
  ```


