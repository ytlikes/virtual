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
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        z-index: 5;
        pointer-events: auto;
    }
    
    /* Esconde só o visual mas mantém clicável */
    div[data-testid="stVerticalBlock"] > div:has(audio) > div {
        opacity: 0;
        width: 200px;
        height: 200px;
        border-radius: 50%;
        cursor: pointer;
    }

    /* Container centralizado */
    .orb-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        min-height: 65vh;
        gap: 2rem;
    }

    /* Orbe wrapper */
    .orb-wrapper {
        position: relative;
        width: 200px;
        height: 200px;
        display: flex;
        justify-content: center;
        align-items: center;
        cursor: pointer;
    }

    /* Núcleo do orbe - ESTADO IDLE */
    .orb-core {
        width: 180px;
        height: 180px;
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
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.8rem;
        animation: fadeIn 0.3s ease;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(5px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .user-message {
        background-color: rgba(56, 189, 248, 0.1);
        border-left: 3px solid #38bdf8;
        text-align: right;
    }

    .bot-message {
        background-color: rgba(31, 41, 55, 0.5);
        border-left: 3px solid rgba(168, 85, 247, 0.5);
    }

    .message-text {
        font-size: 15px;
        line-height: 1.5;
    }

    .message-label {
        font-size: 11px;
        font-weight: 600;
        opacity: 0.6;
        margin-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# 🧠 FUNÇÕES DE BACKEND
# ═══════════════════════════════════════════════════════════

@st.cache_resource
def get_ai_chain():
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.3, max_tokens=100)
    prompt = ChatPromptTemplate.from_template(
        "Você é o MonkeyAI. Responda de forma breve e direta.\nHuman: {input}\nMonkeyAI:"
    )
    return prompt | llm | StrOutputParser()

def transcribe_audio(audio_bytes):
    """Transcreve áudio bytes"""
    r = sr.Recognizer()
    try:
        audio_data = sr.AudioData(audio_bytes, 44100, 2)
        text = r.recognize_google(audio_data, language='pt-BR')
        return text
    except sr.UnknownValueError:
        return None
    except sr.RequestError as e:
        st.error(f"Erro no serviço de reconhecimento: {e}")
        return None
    except Exception as e:
        st.error(f"Erro ao transcrever: {e}")
        return None

def text_to_speech_gtts(text, lang_choice):
    """Gera áudio usando gTTS"""
    try:
        clean_text = re.sub(r'http\S+', 'link', text)
        language = 'pt'
        tld = 'com.br' if lang_choice == 'Brasil' else 'pt'
        
        tts = gTTS(text=clean_text, lang=language, tld=tld, slow=False)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp
    except Exception as e:
        st.warning(f"Erro ao gerar voz: {e}")
        return None

def check_commands(text):
    """Verifica comandos de redirecionamento"""
    text_lower = text.lower()
    
    # YouTube
    if re.search(r'\b(tocar|ouvir|ver|assistir|youtube|vídeo|música)\b', text_lower):
        term = re.sub(
            r'\b(tocar|ouvir|ver|assistir|video|vídeo|musica|música|no|youtube|o|a|de|da|do)\b',
            '', text_lower, flags=re.IGNORECASE
        ).strip()
        
        if term:
            url = f"https://www.youtube.com/results?search_query={term.replace(' ', '+')}"
            return "youtube", url, f"Abrindo YouTube: {term}"
    
    # Google
    if re.search(r'\b(pesquisar|buscar|google|procurar)\b', text_lower):
        term = re.sub(
            r'\b(pesquisar|buscar|google|procurar|sobre|o que é|a|no)\b',
            '', text_lower, flags=re.IGNORECASE
        ).strip()
        
        if term:
            url = f"https://www.google.com/search?q={term.replace(' ', '+')}"
            return "google", url, f"Pesquisando: {term}"
    
    return "chat", None, None

def open_link_js(url):
    """Abre link em nova aba"""
    components.html(f"<script>window.open('{url}', '_blank');</script>", height=0)

# ═══════════════════════════════════════════════════════════
# 📱 INTERFACE PRINCIPAL
# ═══════════════════════════════════════════════════════════
def main():
    # Inicializa estados
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "is_listening" not in st.session_state:
        st.session_state.is_listening = False
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False

    # Barra lateral minimalista
    with st.sidebar:
        st.header("⚙️ Config")
        
        input_mode = st.radio("Modo:", ("🗣️ Voz", "⌨️ Texto"))
        
        if "Voz" in input_mode:
            voice_accent = st.selectbox("Sotaque:", ("Brasil", "Portugal"))
        
        if st.button("🧹 Limpar"):
            st.session_state.messages = []
            st.session_state.is_listening = False
            st.session_state.is_processing = False
            st.rerun()

    # Título
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 40px;'>"
        "MONKEY<span style='color:#38bdf8;'>AI</span></h1>", 
        unsafe_allow_html=True
    )

    user_input = None

    # MODO VOZ
    if "Voz" in input_mode:
        # Define visual do orbe baseado no estado
        orb_class = ""
        orb_icon = ""
        status_text = "Clique no orbe para falar"
        
        if st.session_state.is_listening:
            orb_class = "listening"
            orb_icon = ""
            status_text = "Ouvindo..."
        elif st.session_state.is_processing:
            orb_class = "processing"
            orb_icon = ""
            status_text = "Processando..."
        
        # Container do orbe
        st.markdown(f"""
        <div class="orb-container">
            <div class="orb-wrapper">
                <div class="orb-core {orb_class}">
                    <div class="orb-icon">{orb_icon}</div>
                </div>
            </div>
            <div class="status-text">{status_text}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Gravador de áudio (oculto via CSS)
        if not st.session_state.is_processing:
            audio_bytes = audio_recorder(
                text="",
                recording_color="#ef4444",
                neutral_color="#3b82f6",
                icon_name="microphone",
                icon_size="3x",
                key="audio_recorder"
            )
            
            # Processa quando há áudio
            if audio_bytes and not st.session_state.is_processing:
                st.session_state.is_listening = False
                st.session_state.is_processing = True
                st.rerun()
                
                # Transcreve
                user_input = transcribe_audio(audio_bytes)
                
                if not user_input:
                    st.toast("⚠️ Não consegui entender")
                    st.session_state.is_processing = False
                    st.rerun()
            elif audio_bytes is None and not st.session_state.is_listening and not st.session_state.is_processing:
                # Mostra que está pronto para gravar
                st.session_state.is_listening = False
    
    # MODO TEXTO
    else:
        user_input = st.chat_input("Digite aqui...")

    # PROCESSAMENTO
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        cmd_type, url, reply_text = check_commands(user_input)
        
        if cmd_type != "chat":
            # Comando de redirecionamento
            ai_response = reply_text
            st.session_state.messages.append({"role": "bot", "content": ai_response})
            st.toast(f"🚀 {ai_response}")
            open_link_js(url)
            
            if "Voz" in input_mode:
                audio_fp = text_to_speech_gtts(ai_response, voice_accent)
                if audio_fp:
                    st.audio(audio_fp, format='audio/mp3')
        else:
            # Conversa com IA
            with st.spinner("Pensando..."):
                chain = get_ai_chain()
                ai_response = chain.invoke({"input": user_input})
            
            st.session_state.messages.append({"role": "bot", "content": ai_response})
            
            if "Voz" in input_mode:
                audio_fp = text_to_speech_gtts(ai_response, voice_accent)
                if audio_fp:
                    st.audio(audio_fp, format='audio/mp3')
        
        # Reseta estados
        st.session_state.is_listening = False
        st.session_state.is_processing = False
        st.rerun()

    # HISTÓRICO
    if st.session_state.messages:
        st.divider()
        for msg in st.session_state.messages:
            css = "user-message" if msg["role"] == "user" else "bot-message"
            label = "VOCÊ" if msg["role"] == "user" else "MONKEYAI"
            st.markdown(f"""
                <div class="chat-message {css}">
                    <div class="message-label">{label}</div>
                    <div class="message-text">{msg['content']}</div>
                </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
