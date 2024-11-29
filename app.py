import os
import logging
import asyncio
import tempfile
from typing import Dict, Any

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
import qdrant_client
import edge_tts

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, static_folder='.')
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TTS_VOICE = os.getenv("TTS_VOICE", "id-ID-ArdiNeural")  # Default to Indonesian voice

# Initialize conversation memory with larger window size
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer",
    k=10  # Increased to keep last 10 conversations for better context
)

def get_vector_store() -> Qdrant:
    """Initialize and return the Qdrant vector store."""
    try:
        client = qdrant_client.QdrantClient(
            url=QDRANT_HOST,
            api_key=QDRANT_API_KEY
        )
        logger.info("Qdrant client connected successfully.")
        
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        logger.info("OpenAI embeddings created successfully.")

        vector_store = Qdrant(
            client=client, 
            collection_name=QDRANT_COLLECTION_NAME, 
            embeddings=embeddings,
        )
        return vector_store
    except Exception as e:
        logger.error(f"Error in get_vector_store: {str(e)}")
        raise

def create_qa_chain(vector_store: Qdrant) -> ConversationalRetrievalChain:
    """Create and return the conversational question-answering chain."""
    prompt_template = """
    Anda adalah Klaris, asisten virtual cerdas dan ramah dari Universitas Klabat. Ikuti pedoman berikut:

    1. Berikan jawaban yang singkat, padat, dan informatif.
    2. Gunakan data resmi Universitas Klabat dan nyatakan dengan jelas jika informasi terbatas.
    3. Jawab pertanyaan dengan gaya yang profesional namun bersahabat.
    4. Fokus pada informasi yang diminta tanpa menambahkan detail yang tidak perlu.
    5. Untuk pertanyaan tentang personel (seperti dekan, dosen):
       - Sebutkan nama lengkap dengan gelar akademis
       - Jika ditanya tentang gelar, jelaskan secara spesifik
       - Jika ada komentar positif tentang jawaban sebelumnya, berikan respon yang sesuai konteks
    6. Untuk pertanyaan tentang program studi/jurusan:
       - Berikan informasi lengkap dengan singkatan resmi
       - Sebutkan detail penting seperti akreditasi jika relevan
    7. Jika ada pertanyaan "ulangi", berikan jawaban sebelumnya dengan lebih ringkas dan tepat.
    8. Hindari menambahkan frasa basa-basi seperti "Apakah ada yang ingin ditanyakan lebih lanjut?"
    9. Gunakan bahasa Indonesia yang baik dan benar.
    10. Pastikan setiap jawaban akurat dan terkini.
    11. Pahami konteks percakapan sebelumnya dan berikan respon yang sesuai untuk komentar atau pujian.
    12. Perhatikan alur percakapan dan berikan jawaban yang berkesinambungan dengan topik yang sedang dibahas.

    Riwayat Chat sebelumnya: {chat_history}
    Konteks tambahan: {context}
    Pertanyaan saat ini: {question}
    
    Berikan jawaban yang profesional dan informatif:
    """

    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question", "chat_history"]
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 10})  # Increased for more context

    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model='gpt-4o-mini', temperature=0.7, openai_api_key=OPENAI_API_KEY),
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={'prompt': PROMPT},
        verbose=True
    )

    return qa

async def text_to_speech(text: str) -> str:
    """Convert text to speech using edge-tts."""
    try:
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        communicate = edge_tts.Communicate(text, TTS_VOICE)
        await communicate.save(output_file.name)
        logger.info(f"Audio file generated: {output_file.name}")
        return output_file.name
    except Exception as e:
        logger.error(f"Error in text_to_speech: {str(e)}")
        raise

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/video/<path:filename>')
def serve_video(filename):
    return send_from_directory('video', filename)

@app.route('/process-speech', methods=['POST'])
def process_speech():
    data = request.json
    text = data.get('text')

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        vector_store = get_vector_store()
        qa_chain = create_qa_chain(vector_store)

        logger.info(f"Processing query: {text}")
        logger.info(f"Current chat history: {memory.chat_memory.messages}")
        
        result = qa_chain({"question": text})

        answer = result['answer']
        logger.info(f"Answer generated: {answer[:50]}...")

        audio_file = asyncio.run(text_to_speech(answer))
        audio_filename = os.path.basename(audio_file)

        return jsonify({
            "text": answer,
            "audioUrl": f"/api/audio/{audio_filename}"
        })
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({"error": "Terjadi kesalahan: " + str(e)}), 500

@app.route('/api/audio/<path:filename>', methods=['GET'])
def serve_audio(filename):
    try:
        directory = tempfile.gettempdir()
        return send_file(os.path.join(directory, filename), mimetype="audio/mpeg")
    except Exception as e:
        logger.error(f"Error serving audio file: {str(e)}")
        return jsonify({"error": "Terjadi kesalahan saat menyajikan file audio"}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint tidak ditemukan"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Terjadi kesalahan internal server"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)