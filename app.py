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
# 🎨 CSS MINIMALISTA E CLEAN
# ═══════════════════════════════════════════════════════════
st.markdown("""
<style>
    .stApp {
        background-color: #000000;
        background-image: radial-gradient(circle at center, #0a0e1a 0%, #000000 100%);
        color: #e0f2fe;
    }
    
    #MainMenu, footer {visibility: hidden;}
    .stDeployButton {display:none;}

    /* Esconde completamente o recorder padrão */
    div[data-testid="stVerticalBlock"] > div:has(audio) {
        position: absolute;
        top: 40%;
        left: 50%;
        transform: translate(-50%, -50%);
        z-index: 15;
        pointer-events: auto;
        width: 250px !important;
    }
    
    /* Botão de gravação visível e grande para mobile */
    div[data-testid="stVerticalBlock"] > div:has(audio) > div {
        opacity: 1 !important;
        width: 100% !important;
        height: 80px !important;
        border-radius: 12px !important;
        cursor: pointer;
        background: rgba(56, 189, 248, 0.1) !important;
        border: 2px solid rgba(56, 189, 248, 0.3) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 16px !important;
        color: #38bdf8 !important;
    }
    
    div[data-testid="stVerticalBlock"] > div:has(audio) button {
        font-size: 18px !important;
        padding: 1rem 2rem !important;
    }

    /* Container do Orbe - Altura ajustada para caber o chat embaixo */
    .orb-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        min-height: 350px; /* Reduzido para aproximar o chat */
        margin-bottom: 20px;
        gap: 2rem;
    }

    /* Orbe wrapper */
    .orb-wrapper {
        position: relative;
        width: 180px;
        height: 180px;
        display: flex;
        justify-content: center;
        align-items: center;
        cursor: pointer;
    }

    /* Núcleo do orbe - ESTADO IDLE */
    .orb-core {
        width: 160px;
        height: 160px;
        border-radius: 50%;
        background: radial-gradient(circle at 30% 30%, 
            rgba(56, 189, 248, 0.4), 
            rgba(3, 105, 161, 0.7) 50%, 
            rgba(0, 20, 40, 1) 100%);
        box-shadow: 
            0 0 20px rgba(56, 189, 248, 0.3),
            0 0 40px rgba(3, 105, 161, 0.2),
            inset 0 0 30px rgba(255, 255, 255, 0.1);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        display: flex;
        justify-content: center;
        align-items: center;
        animation: idle-pulse 4s infinite ease-in-out;
    }

    /* Hover */
    .orb-wrapper:hover .orb-core {
        transform: scale(1.05);
        box-shadow: 
            0 0 40px rgba(56, 189, 248, 0.6),
            0 0 80px rgba(3, 105, 161, 0.4),
            inset 0 0 40px rgba(255, 255, 255, 0.2);
    }

    /* ESTADO OUVINDO - vermelho pulsante */
    .orb-core.listening {
        background: radial-gradient(circle at 30% 30%, 
            rgba(239, 68, 68, 0.8), 
            rgba(185, 28, 28, 1) 50%, 
            rgba(80, 10, 10, 1) 100%);
        box-shadow: 
            0 0 40px rgba(239, 68, 68, 0.8),
            0 0 80px rgba(185, 28, 28, 0.6),
            0 0 120px rgba(153, 27, 27, 0.4),
            inset 0 0 40px rgba(255, 100, 100, 0.3);
        animation: listening-pulse 1.2s infinite ease-in-out;
    }

    /* ESTADO PROCESSANDO - roxo girando */
    .orb-core.processing {
        background: radial-gradient(circle at 30% 30%, 
            rgba(168, 85, 247, 0.8), 
            rgba(126, 34, 206, 1) 50%, 
            rgba(60, 10, 80, 1) 100%);
        box-shadow: 
            0 0 40px rgba(168, 85, 247, 0.8),
            0 0 80px rgba(126, 34, 206, 0.6),
            inset 0 0 40px rgba(200, 150, 255, 0.3);
        animation: processing-spin 2s infinite linear;
    }

    @keyframes idle-pulse {
        0%, 100% { opacity: 0.85; }
        50% { opacity: 1; }
    }

    @keyframes listening-pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.08); }
    }

    @keyframes processing-spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    .orb-icon {
        font-size: 60px;
        filter: drop-shadow(0 0 10px rgba(255, 255, 255, 0.5));
        pointer-events: none;
    }

    /* Status text minimalista */
    .status-text {
        text-align: center;
        font-size: 14px;
        opacity: 0.6;
        letter-spacing: 1px;
        min-height: 20px;
    }

    /* Esconde o botão de gravação padrão */
    div[data-testid="stHorizontalBlock"] > div:has(audio) {
        display: none !important;
    }

    /* Chat messages */
    .chat-container {
        margin-top: 20px;
        display: flex;
        flex-direction: column;
        gap: 15px;
    }

    .chat-message {
        padding: 1rem;
        border-radius: 12px;
        animation: fadeIn 0.4s ease;
        position: relative;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .user-message {
        background-color: rgba(56, 189, 248, 0.15);
        border: 1px solid rgba(56, 189, 248, 0.3);
        text-align: right;
        align-self: flex-end;
        margin-left: 20%;
    }

    .bot-message {
        background-color: rgba(31, 41, 55, 0.6);
        border: 1px solid rgba(168, 85, 247, 0.3);
        text-align: left;
        align-self: flex-start;
        margin-right: 20%;
    }

    .message-text {
        font-size: 16px;
        line-height: 1.6;
        color: #e0f2fe;
    }

    .message-label {
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.5;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# 🧠 FUNÇÕES DE BACKEND
# ═══════════════════════════════════════════════════════════

@st.cache_resource
def get_ai_chain():
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.3, max_tokens=100)
    prompt = ChatPromptTemplate.from_template(
        "Você é um assistente útil e direto. Responda em português.\nPergunta: {input}\nResposta:"
    )
    return prompt | llm | StrOutputParser()

def transcribe_audio(audio_bytes):
    """Transcreve áudio bytes tratando como arquivo WAV"""
    if not audio_bytes:
        return None
    
    audio_file = io.BytesIO(audio_bytes)
    r = sr.Recognizer()
    r.energy_threshold = 300
    r.pause_threshold = 0.5
    
    try:
        with sr.AudioFile(audio_file) as source:
            audio_data = r.record(source)
        text = r.recognize_google(audio_data, language='pt-BR')
        return text.strip() if text else None
    except Exception:
        return None

def text_to_speech_gtts(text, lang_choice):
    """Gera áudio usando gTTS"""
    try:
        clean_text = re.sub(r'http\S+', 'link', text)
        if len(clean_text) > 250:
            clean_text = clean_text[:250] + "..."
        
        language = 'pt'
        tld = 'com.br' if lang_choice == 'Brasil' else 'pt'
        
        tts = gTTS(text=clean_text, lang=language, tld=tld, slow=False)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp
    except Exception:
        return None

def check_commands(text):
    text_lower = text.lower()
    
    if re.search(r'\b(tocar|ouvir|ver|assistir|youtube|vídeo)\b', text_lower):
        term = re.sub(r'\b(tocar|ouvir|ver|assistir|video|vídeo|musica|música|no|youtube|o|a|de|da|do)\b', '', text_lower, flags=re.IGNORECASE).strip()
        if term:
            return "youtube", f"https://www.youtube.com/results?search_query={term.replace(' ', '+')}", f"Abrindo YouTube: {term}"
    
    if re.search(r'\b(pesquisar|buscar|google|procurar)\b', text_lower):
        term = re.sub(r'\b(pesquisar|buscar|google|procurar|sobre|o que é|a|no)\b', '', text_lower, flags=re.IGNORECASE).strip()
        if term:
            return "google", f"https://www.google.com/search?q={term.replace(' ', '+')}", f"Pesquisando: {term}"
    
    return "chat", None, None

def open_link_js(url):
    components.html(f"<script>window.open('{url}', '_blank');</script>", height=0)

# ═══════════════════════════════════════════════════════════
# 📱 INTERFACE PRINCIPAL
# ═══════════════════════════════════════════════════════════
def main():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Config")
        input_mode = st.radio("Modo:", ("🗣️ Voz", "⌨️ Texto"))
        voice_accent = st.selectbox("Sotaque:", ("Brasil", "Portugal")) if "Voz" in input_mode else "Brasil"
        if st.button("🧹 Limpar Chat"):
            st.session_state.messages = []
            st.session_state.is_processing = False
            st.rerun()

    # Título
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 20px;'>"
        "MONKEY<span style='color:#38bdf8;'>AI</span></h1>", 
        unsafe_allow_html=True
    )

    # ─────────────────────────────────────────────────────────
    # ÁREA DE INPUT (VOZ OU TEXTO)
    # ─────────────────────────────────────────────────────────
    
    # Placeholder para interação
    new_user_input = None

    if "Voz" in input_mode:
        # Visual do Orbe
        orb_class = "processing" if st.session_state.is_processing else ""
        status_text = "Processando..." if st.session_state.is_processing else "Toque para falar"
        
        st.markdown(f"""
        <div class="orb-container">
            <div class="orb-wrapper">
                <div class="orb-core {orb_class}">
                    <div class="orb-icon"></div>
                </div>
            </div>
            <div class="status-text">{status_text}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Recorder (só mostra se não estiver processando)
        audio_bytes = None
        if not st.session_state.is_processing:
            audio_bytes = audio_recorder(
                text="", # Texto vazio para limpar visual
                recording_color="#ef4444",
                neutral_color="#3b82f6",
                icon_name="microphone",
                icon_size="2x",
                key="audio_recorder",
                pause_threshold=2.0
            )
        
        # Lógica de processamento de Voz
        if audio_bytes and len(audio_bytes) > 2000 and not st.session_state.is_processing:
            st.session_state.is_processing = True
            text_transcribed = transcribe_audio(audio_bytes)
            
            if text_transcribed:
                new_user_input = text_transcribed
            else:
                st.toast("⚠️ Não entendi")
                st.session_state.is_processing = False
                st.rerun()

    else:
        # Modo Texto
        new_user_input = st.chat_input("Digite sua mensagem...")

    # ─────────────────────────────────────────────────────────
    # PROCESSAMENTO CENTRALIZADO (VOZ E TEXTO)
    # ─────────────────────────────────────────────────────────
    if new_user_input:
        # 1. Adiciona pergunta do usuário
        st.session_state.messages.append({"role": "user", "content": new_user_input})
        
        # 2. Verifica comandos
        cmd_type, url, reply_text = check_commands(new_user_input)
        
        if cmd_type != "chat":
            # Resposta de comando
            st.session_state.messages.append({"role": "bot", "content": reply_text})
            open_link_js(url)
        else:
            # Resposta da IA
            chain = get_ai_chain()
            ai_response = chain.invoke({"input": new_user_input})
            st.session_state.messages.append({"role": "bot", "content": ai_response})
            
            # Se for modo voz, gera áudio
            if "Voz" in input_mode:
                audio_fp = text_to_speech_gtts(ai_response, voice_accent)
                if audio_fp:
                    st.session_state['last_audio'] = audio_fp

        st.session_state.is_processing = False
        st.rerun()

    # Toca áudio se houver (Auto-play)
    if 'last_audio' in st.session_state and st.session_state.last_audio:
        st.audio(st.session_state.last_audio, format='audio/mp3', autoplay=True)
        st.session_state.last_audio = None

    # ─────────────────────────────────────────────────────────
    # VISUALIZAÇÃO DO CHAT (SEMPRE ABAIXO DO INPUT DE VOZ)
    # ─────────────────────────────────────────────────────────
    if st.session_state.messages:
        # Container HTML para o chat
        chat_html = '<div class="chat-container">'
        
        # Inverte a lista para mostrar as mais recentes no topo (opcional, aqui mantive padrão)
        # Se quiser estilo WhatsApp (novas embaixo), mantenha a ordem normal.
        for msg in st.session_state.messages:
            role_class = "user-message" if msg["role"] == "user" else "bot-message"
            label = "VOCÊ" if msg["role"] == "user" else "MONKEYAI"
            
            chat_html += f"""
                <div class="chat-message {role_class}">
                    <div class="message-label">{label}</div>
                    <div class="message-text">{msg['content']}</div>
                </div>
            """
        
        chat_html += '</div>'
        st.markdown(chat_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
