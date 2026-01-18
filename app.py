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
# ⚙️ CONFIGURAÇÃO INICIAL
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MonkeyAI",
    page_icon="🐵",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Carrega API Key
load_dotenv()
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

if not os.getenv("GROQ_API_KEY"):
    st.error("⚠️ Erro: Chave API Groq não configurada no .env ou secrets.")
    st.stop()

# ═══════════════════════════════════════════════════════════
# 🎨 CSS COMPLETO (ESTILO VISUAL)
# ═══════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* Fundo e cores gerais */
    .stApp {
        background-color: #000000;
        background-image: radial-gradient(circle at center, #0a0e1a 0%, #000000 100%);
        color: #e0f2fe;
    }
    
    /* Remove elementos padrão do Streamlit */
    #MainMenu, footer, .stDeployButton {visibility: hidden;}

    /* ─── ESTILOS DO GRAVADOR (ESCONDIDO/ESTILIZADO) ─── */
    div[data-testid="stVerticalBlock"] > div:has(audio) {
        position: absolute;
        top: 35%; 
        left: 50%;
        transform: translate(-50%, -50%);
        z-index: 15;
        width: 200px !important;
        pointer-events: auto;
    }
    
    /* Botão de gravação transparente sobre o orbe */
    div[data-testid="stVerticalBlock"] > div:has(audio) > div {
        opacity: 0.01 !important; /* Quase invisível mas clicável */
        width: 100% !important;
        height: 180px !important;
        border-radius: 50% !important;
        cursor: pointer;
    }

    /* ─── ESTILOS DO ORBE (ANIMAÇÃO) ─── */
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

    .orb-core.listening {
        background: radial-gradient(circle at 30% 30%, rgba(239, 68, 68, 0.8), rgba(185, 28, 28, 1) 50%, rgba(80, 10, 10, 1) 100%);
        box-shadow: 0 0 50px rgba(239, 68, 68, 0.6);
        animation: listening-pulse 1.2s infinite ease-in-out;
    }

    .orb-core.processing {
        background: radial-gradient(circle at 30% 30%, rgba(168, 85, 247, 0.8), rgba(126, 34, 206, 1) 50%, rgba(60, 10, 80, 1) 100%);
        box-shadow: 0 0 50px rgba(168, 85, 247, 0.6);
        animation: processing-spin 2s infinite linear;
    }

    @keyframes idle-pulse { 0%, 100% { opacity: 0.8; } 50% { opacity: 1; } }
    @keyframes listening-pulse { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.1); } }
    @keyframes processing-spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }

    .status-text {
        margin-top: 15px;
        font-size: 14px;
        letter-spacing: 1px;
        opacity: 0.7;
    }

    /* ─── ESTILOS DO CHAT UNIFICADO ─── */
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 15px;
        padding: 10px;
        margin-top: 10px;
        border-top: 1px solid rgba(255,255,255,0.1);
    }

    .chat-row {
        display: flex;
        width: 100%;
    }

    .row-user { justify-content: flex-end; }
    .row-bot { justify-content: flex-start; }

    .chat-bubble {
        max-width: 80%;
        padding: 12px 16px;
        border-radius: 12px;
        font-size: 15px;
        line-height: 1.5;
        position: relative;
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
    
    .bubble-label {
        font-size: 10px;
        opacity: 0.5;
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# 🧠 FUNÇÕES DE BACKEND
# ═══════════════════════════════════════════════════════════

@st.cache_resource
def get_ai_chain():
    """Configura o modelo Llama via Groq"""
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.3, max_tokens=150)
    prompt = ChatPromptTemplate.from_template(
        "Você é o MonkeyAI, um assistente inteligente. Responda em português de forma concisa.\nPergunta: {input}\nResposta:"
    )
    return prompt | llm | StrOutputParser()

def transcribe_audio(audio_bytes):
    """Transcreve áudio bytes (CORRIGIDO PARA ARQUIVOS WAV)"""
    if not audio_bytes:
        return None
    
    # IMPORTANTE: Converte bytes crus em arquivo virtual
    audio_file = io.BytesIO(audio_bytes)
    r = sr.Recognizer()
    
    try:
        with sr.AudioFile(audio_file) as source:
            audio_data = r.record(source)
        
        text = r.recognize_google(audio_data, language='pt-BR')
        return text.strip() if text else None
    except Exception:
        return None

def text_to_speech_gtts(text):
    """Gera áudio usando gTTS"""
    try:
        # Limpa texto para evitar leitura de links longos
        clean_text = re.sub(r'http\S+', 'um link', text)
        if len(clean_text) > 300:
            clean_text = clean_text[:300]
        
        tts = gTTS(text=clean_text, lang='pt', tld='com.br', slow=False)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp
    except:
        return None

def check_commands(text):
    """Verifica comandos de sistema (Youtube/Google)"""
    text_lower = text.lower()
    
    if re.search(r'\b(youtube|vídeo)\b', text_lower):
        term = re.sub(r'\b(youtube|vídeo|assistir|ver|tocar|ouvir)\b', '', text_lower, flags=re.IGNORECASE).strip()
        if term:
            return "youtube", f"https://www.youtube.com/results?search_query={term.replace(' ', '+')}", f"Abrindo YouTube para: {term}"
    
    if re.search(r'\b(google|pesquisar|buscar)\b', text_lower):
        term = re.sub(r'\b(google|pesquisar|buscar|procurar|o que é)\b', '', text_lower, flags=re.IGNORECASE).strip()
        if term:
            return "google", f"https://www.google.com/search?q={term.replace(' ', '+')}", f"Pesquisando no Google: {term}"
            
    return "chat", None, None

def open_link_js(url):
    components.html(f"<script>window.open('{url}', '_blank');</script>", height=0)

# ═══════════════════════════════════════════════════════════
# 📱 LÓGICA PRINCIPAL DA INTERFACE
# ═══════════════════════════════════════════════════════════
def main():
    # Inicializa Estados
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False

    # Sidebar Config
    with st.sidebar:
        st.header("MonkeyAI Config")
        input_mode = st.radio("Modo de Entrada:", ("🗣️ Voz", "⌨️ Texto"))
        if st.button("Limpar Conversa"):
            st.session_state.messages = []
            st.rerun()

    # Título
    st.markdown("<h1 style='text-align: center;'>MONKEY<span style='color:#38bdf8;'>AI</span></h1>", unsafe_allow_html=True)

    # Variável para armazenar o novo input (seja voz ou texto)
    new_input = None

    # ─────────────────────────────────────────────────────────
    # 1. ZONA DE INPUT (VOZ NO TOPO OU TEXTO EMBAIXO)
    # ─────────────────────────────────────────────────────────
    
    if "Voz" in input_mode:
        # Visual do Orbe
        orb_status = "processing" if st.session_state.is_processing else ""
        label_status = "Processando..." if st.session_state.is_processing else "Toque no orbe para falar"
        
        st.markdown(f"""
        <div class="orb-container">
            <div class="orb-wrapper">
                <div class="orb-core {orb_status}"></div>
            </div>
            <div class="status-text">{label_status}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # O Gravador (Invisível, mas funcional sobre o orbe)
        audio_bytes = audio_recorder(
            text="", 
            recording_color="#ef4444", 
            neutral_color="#3b82f6", 
            icon_size="1x",
            key="voice_rec",
            pause_threshold=2.0
        )
        
        # Processamento do Áudio
        if audio_bytes and len(audio_bytes) > 2000 and not st.session_state.is_processing:
            st.session_state.is_processing = True
            transcription = transcribe_audio(audio_bytes)
            
            if transcription:
                new_input = transcription
            else:
                st.toast("⚠️ Não entendi o áudio")
                st.session_state.is_processing = False
                st.rerun()

    # ─────────────────────────────────────────────────────────
    # 2. RENDERIZAÇÃO DO CHAT (HISTÓRICO)
    # ─────────────────────────────────────────────────────────
    
    # Container do Chat (Sempre visível abaixo do orbe)
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    for msg in st.session_state.messages:
        row_class = "row-user" if msg["role"] == "user" else "row-bot"
        bubble_class = "bubble-user" if msg["role"] == "user" else "bubble-bot"
        label = "VOCÊ" if msg["role"] == "user" else "MONKEY AI"
        
        st.markdown(f"""
            <div class="chat-row {row_class}">
                <div class="chat-bubble {bubble_class}">
                    <div class="bubble-label">{label}</div>
                    {msg['content']}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────
    # 3. INPUT DE TEXTO (SELECIONADO)
    # ─────────────────────────────────────────────────────────
    if "Texto" in input_mode:
        text_input = st.chat_input("Digite sua mensagem...")
        if text_input:
            new_input = text_input

    # ─────────────────────────────────────────────────────────
    # 4. LÓGICA CENTRAL (PROCESSAR COMANDOS/IA)
    # ─────────────────────────────────────────────────────────
    if new_input:
        # Adiciona pergunta do usuário
        st.session_state.messages.append({"role": "user", "content": new_input})
        
        # Verifica Comandos
        cmd, url, response_text = check_commands(new_input)
        
        if cmd != "chat":
            # Resposta de Comando
            st.session_state.messages.append({"role": "bot", "content": response_text})
            open_link_js(url)
        else:
            # Resposta da IA
            chain = get_ai_chain()
            ai_reply = chain.invoke({"input": new_input})
            st.session_state.messages.append({"role": "bot", "content": ai_reply})
            
            # Gera áudio apenas se estiver no modo voz
            if "Voz" in input_mode:
                audio_fp = text_to_speech_gtts(ai_reply)
                if audio_fp:
                    st.session_state['last_audio'] = audio_fp

        st.session_state.is_processing = False
        st.rerun()

    # ─────────────────────────────────────────────────────────
    # 5. AUTO-PLAY ÁUDIO
    # ─────────────────────────────────────────────────────────
    if 'last_audio' in st.session_state and st.session_state.last_audio:
        st.audio(st.session_state.last_audio, format='audio/mp3', autoplay=True)
        st.session_state.last_audio = None

if __name__ == "__main__":
    main()
