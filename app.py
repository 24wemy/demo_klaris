import os
import logging
import asyncio
import tempfile
from typing import Dict, Any

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
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

def create_qa_chain(vector_store: Qdrant) -> RetrievalQA:
    """Create and return the question-answering chain."""
    prompt_template = """
    Anda adalah Klaris, asisten virtual Universitas Klabat. Ikuti pedoman berikut:

    1. Jawab pertanyaan dengan singkat namun akurat, maksimal 200 karakter.
    2. Gunakan data resmi Universitas Klabat. Jika tidak yakin, nyatakan keterbatasan informasi.
    3. Tangani pertanyaan ganda dengan cermat, menjawab setiap bagian.
    4. Prioritaskan informasi terkini tentang akademik, admisi, dan kehidupan kampus.
    5. Gunakan bahasa formal dan sopan, sesuai budaya akademik.
    6. Tawarkan bantuan lebih lanjut atau rujukan ke sumber resmi jika diperlukan.
    7. Sesuaikan respons dengan konteks pertanyaan (mahasiswa, calon mahasiswa, dll).

    *jawab sesuai dengan apa yang ditanya saja, contoh pertanyaan:
    apa saja matakuliah semeseter 1 jurusan informatika:
    jawaban: sebutkan nama matakuliahnya tanpa prerequiset matakuliahnya.

    konteks: {context}
    pertanyaan: {question}
    jawaban:
    """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain_type_kwargs = {"prompt": PROMPT}
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})

    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model='gpt-4o-mini', temperature=0.7, openai_api_key=OPENAI_API_KEY),
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True
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
        result = qa_chain({"query": text})

        answer = result['result']
        logger.info(f"Answer generated: {answer[:50]}...")  # Log first 50 characters

        # Generate TTS
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