# 📚 Conversational RAG with PDF Uploads and Chat History

This project implements a **Conversational Retrieval-Augmented Generation (RAG)** system where users can upload PDFs, ask questions, and get answers based on document content — all with **persistent chat history**. Built with **LangChain**, **Streamlit**, and **FAISS**, it allows seamless knowledge extraction from documents using LLMs.

---

## 🚀 Features

- 📁 **Upload PDFs** and extract context automatically
- 🤖 Ask natural questions and get answers grounded in your uploaded docs
- 🧠 Built with **LangChain**, **FAISS**, and **LLMs**
- 💬 **Chat history** retained for every session
- 🖥️ **Streamlit UI** for clean and easy interaction

---

## 🧰 Tech Stack

- **Python 3.10+**
- **LangChain**
- **FAISS**
- **PyPDF2**
- **Streamlit**
- **Groq LLM (ChatGroq)**
- **dotenv** for secure key management

---

## 🧑‍💻 Installation

**1. Clone the repository**
- git clone https://github.com/Kuntalsvyas/Conversational-RAG-with-PDF-uploads-and-chat-history.git
- cd Conversational-RAG-with-PDF-uploads-and-chat-history

**2. Create a virtual environment**
- python -m venv venv
- source venv/bin/activate  # For Windows: venv\Scripts\activate

**3. Install dependencies**
- pip install -r requirements.txt

**4. Add your environment variables in a `.env` file**
- Example :- GROQ_API_KEY=your_groq_api_key


**5. Run the Streamlit app**
- streamlit run app.py

---

# 🧪 How It Works
- Upload a PDF → Text is extracted and split into chunks.
- Vectorization → FAISS indexes the chunks using embeddings.
- Chat Input → Your query is compared to chunks for context.
- RAG Pipeline → LLM generates a contextual response.
- Chat History → Stored locally for that session.

---

# 📌 Future Enhancements
 - Add authentication (login system)
 - Save history to cloud or DB
 - Upload multiple PDFs and merge knowledge
 - Add support for YouTube/video transcript ingestion

---

# 🙌 Author
Made with 💻 by Kuntal Vyas
If you find this helpful, ⭐ the repo and share!
