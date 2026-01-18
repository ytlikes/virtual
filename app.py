import os
import io
import re
import streamlit as st
import streamlit.components.v1 as components
from gtts import gTTS
import speech_recognition as sr
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from audio_recorder_streamlit import audio_recorder

# ═══════════════════════════════════════════════════════════
# ⚙️ CONFIGURAÇÃO
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MonkeyAI",
    page_icon="🐵",
    layout="centered",
    initial_sidebar_state="collapsed"
)

load_dotenv()
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

if not os.getenv("GROQ_API_KEY"):
    st.error("⚠️ Erro: Chave API Groq não configurada.")
    st.stop()

# ═══════════════════════════════════════════════════════════
# 🎨 CSS (Visual Ajustado)
# ═══════════════════════════════════════════════════════════
st.markdown("""
<style>
    .stApp {
        background-color: #000000;
        background-image: radial-gradient(circle at center, #0a0e1a 0%, #000000 100%);
        color: #e0f2fe;
    }
    #MainMenu, footer, .stDeployButton {visibility: hidden;}

    /* Posicionamento do gravador (invisível sobre o orbe) */
    div[data-testid="stVerticalBlock"] > div:has(audio) {
        position: absolute;
        top: 35%; 
        left: 50%;
        transform: translate(-50%, -50%);
        z-index: 15;
        width: 180px !important;
        pointer-events: auto;
    }
    div[data-testid="stVerticalBlock"] > div:has(audio) > div {
        opacity: 0.01 !important;
        width: 100% !important;
        height: 180px !important;
        border-radius: 50% !important;
        cursor: pointer;
    }

    /* Orbe */
    .orb-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        padding: 20px 0;
        margin-bottom: 20px;
        min-height: 250px;
    }
    .orb-wrapper {
        position: relative;
        width: 160px;
        height: 160px;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .orb-core {
        width: 140px;
        height: 140px;
        border-radius: 50%;
        background: radial-gradient(circle at 30% 30%, rgba(56, 189, 248, 0.4), rgba(3, 105, 161, 0.7) 50%, rgba(0, 20, 40, 1) 100%);
        box-shadow: 0 0 30px rgba(56, 189, 248, 0.3), inset 0 0 20px rgba(255, 255, 255, 0.1);
        transition: all 0.4s ease;
        animation: idle-pulse 4s infinite ease-in-out;
    }
    .orb-core.processing {
        background: radial-gradient(circle at 30% 30%, rgba(168, 85, 247, 0.8), rgba(126, 34, 206, 1) 50%, rgba(60, 10, 80, 1) 100%);
        box-shadow: 0 0 50px rgba(168, 85, 247, 0.6);
        animation: processing-spin 2s infinite linear;
    }
    
    @keyframes idle-pulse { 0%, 100% { opacity: 0.8; } 50% { opacity: 1; } }
    @keyframes processing-spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }

    .status-text { margin-top: 15px; font-size: 14px; opacity: 0.7; }

    /* Chat */
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 15px;
        padding: 10px;
        margin-top: 10px;
        border-top: 1px solid rgba(255,255,255,0.1);
    }
    .chat-row { display: flex; width: 100%; margin-bottom: 10px; }
    .row-user { justify-content: flex-end; }
    .row-bot { justify-content: flex-start; }
    
    .chat-bubble {
        max-width: 80%;
        padding: 12px 16px;
        border-radius: 12px;
        font-size: 15px;
        line-height: 1.5;
    }
    .bubble-user {
        background-color: rgba(56, 189, 248, 0.15);
        border: 1px solid rgba(56, 189, 248, 0.3);
        color: #e0f2fe;
        border-bottom-right-radius: 2px;
    }
    .bubble-bot {
        background-color: rgba(31, 41, 55, 0.7);
        border: 1px solid rgba(168, 85, 247, 0.3);
        color: #d8b4fe;
        border-bottom-left-radius: 2px;
    }
    .bubble-label { font-size: 10px; opacity: 0.5; margin-bottom: 4px; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# 🧠 FUNÇÕES DE BACKEND
# ═══════════════════════════════════════════════════════════

@st.cache_resource
def get_ai_chain():
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.3, max_tokens=150)
    prompt = ChatPromptTemplate.from_template(
        "Você é o MonkeyAI. Responda em português de forma curta e prestativa.\nPergunta: {input}\nResposta:"
    )
    return prompt | llm | StrOutputParser()

def transcribe_audio(audio_bytes):
    if not audio_bytes: return None
    audio_file = io.BytesIO(audio_bytes)
    r = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio_data = r.record(source)
        text = r.recognize_google(audio_data, language='pt-BR')
        return text.strip() if text else None
    except: return None

def text_to_speech_gtts(text):
    try:
        clean = re.sub(r'http\S+', 'link', text)[:300]
        tts = gTTS(text=clean, lang='pt', tld='com.br', slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp
    except: return None

def check_commands(text):
    text = text.lower()
    if re.search(r'\b(youtube|vídeo)\b', text):
        term = re.sub(r'\b(youtube|vídeo|assistir|ver|tocar)\b', '', text).strip()
        if term: return "youtube", f"https://www.youtube.com/results?search_query={term.replace(' ', '+')}", f"YouTube: {term}"
    if re.search(r'\b(google|pesquisar)\b', text):
        term = re.sub(r'\b(google|pesquisar|buscar)\b', '', text).strip()
        if term: return "google", f"https://www.google.com/search?q={term.replace(' ', '+')}", f"Google: {term}"
    return "chat", None, None

def open_link_js(url):
    components.html(f"<script>window.open('{url}', '_blank');</script>", height=0)

# ═══════════════════════════════════════════════════════════
# 📱 LÓGICA PRINCIPAL
# ═══════════════════════════════════════════════════════════
def main():
    # Inicializa variáveis de estado
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False
    if "last_processed_audio" not in st.session_state:
        st.session_state.last_processed_audio = b"" # Armazena bytes do último áudio processado

    # Sidebar
    with st.sidebar:
        st.header("Config")
        input_mode = st.radio("Modo:", ("🗣️ Voz", "⌨️ Texto"))
        if st.button("Limpar"):
            st.session_state.messages = []
            st.session_state.last_processed_audio = b""
            st.rerun()

    # Título
    st.markdown("<h1 style='text-align: center;'>MONKEY<span style='color:#38bdf8;'>AI</span></h1>", unsafe_allow_html=True)

    # Variável para capturar nova entrada
    new_input = None

    # 1. ÁREA DO ORBE / GRAVADOR
    if "Voz" in input_mode:
        orb_status = "processing" if st.session_state.is_processing else ""
        label_status = "Processando..." if st.session_state.is_processing else "Toque para falar"
        
        st.markdown(f"""
        <div class="orb-container">
            <div class="orb-wrapper">
                <div class="orb-core {orb_status}"></div>
            </div>
            <div class="status-text">{label_status}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Gravador
        audio_bytes = audio_recorder(
            text="", recording_color="#ef4444", neutral_color="#3b82f6", 
            icon_size="1x", key="voice_rec", pause_threshold=2.0
        )
        
        # LÓGICA ANTI-REPETIÇÃO E PROCESSAMENTO
        # Verifica se tem áudio E se ele é diferente do último processado
        if audio_bytes and len(audio_bytes) > 2000 and audio_bytes != st.session_state.last_processed_audio:
            st.session_state.is_processing = True
            
            # 1. Marca como processado para não repetir no loop
            st.session_state.last_processed_audio = audio_bytes
            
            # 2. Transcreve
            transcription = transcribe_audio(audio_bytes)
            
            if transcription:
                new_input = transcription
            else:
                st.toast("⚠️ Áudio não compreendido")
                st.session_state.is_processing = False
                st.rerun()

    # 2. INPUT TEXTO
    if "Texto" in input_mode:
        text_input = st.chat_input("Mensagem...")
        if text_input:
            new_input = text_input

    # 3. PROCESSAMENTO DE ENTRADA (CHAT)
    if new_input:
        # Adiciona pergunta do usuário
        st.session_state.messages.append({"role": "user", "content": new_input})
        
        cmd, url, resp = check_commands(new_input)
        
        if cmd != "chat":
            st.session_state.messages.append({"role": "bot", "content": resp})
            open_link_js(url)
        else:
            chain = get_ai_chain()
            ai_reply = chain.invoke({"input": new_input})
            st.session_state.messages.append({"role": "bot", "content": ai_reply})
            
            if "Voz" in input_mode:
                audio_fp = text_to_speech_gtts(ai_reply)
                if audio_fp:
                    st.session_state['last_audio'] = audio_fp

        st.session_state.is_processing = False
        st.rerun() # FORÇA ATUALIZAÇÃO IMEDIATA DA TELA

    # 4. EXIBIÇÃO DO CHAT (Agora processado antes, então aparece atualizado)
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        r_cls = "row-user" if msg["role"] == "user" else "row-bot"
        b_cls = "bubble-user" if msg["role"] == "user" else "bubble-bot"
        lbl = "VOCÊ" if msg["role"] == "user" else "MONKEY AI"
        
        st.markdown(f"""
            <div class="chat-row {r_cls}">
                <div class="chat-bubble {b_cls}">
                    <div class="bubble-label">{lbl}</div>
                    {msg['content']}
                </div>
            </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 5. PLAYBACK DE ÁUDIO (Auto-play)
    if 'last_audio' in st.session_state and st.session_state.last_audio:
        st.audio(st.session_state.last_audio, format='audio/mp3', autoplay=True)
        st.session_state.last_audio = None

if __name__ == "__main__":
    main()
