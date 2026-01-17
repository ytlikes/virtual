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
# Tenta pegar do secrets ou do .env
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

if not os.getenv("GROQ_API_KEY"):
    st.error("⚠️ Erro: Chave API Groq não configurada.")
    st.stop()

# ═══════════════════════════════════════════════════════════
# 🎨 CSS (Mantido o seu estilo)
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

    /* Centraliza o recorder e deixa invisível sobre o orbe */
    div[data-testid="stVerticalBlock"] > div:has(audio) {
        position: relative;
        z-index: 5;
        opacity: 0.05; /* Deixei levemente visível para debug (mude para 0 depois) */
        height: 180px;
        width: 180px;
        margin: auto;
    }
    
    .orb-container {
        position: absolute;
        top: 20%;
        left: 0;
        right: 0;
        margin: auto;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        pointer-events: none; /* Deixa o clique passar para o recorder */
        z-index: 1;
    }

    .orb-core {
        width: 180px;
        height: 180px;
        border-radius: 50%;
        background: radial-gradient(circle at 30% 30%, rgba(56, 189, 248, 0.4), rgba(3, 105, 161, 0.7) 50%, rgba(0, 20, 40, 1) 100%);
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.3), 0 0 40px rgba(3, 105, 161, 0.2), inset 0 0 30px rgba(255, 255, 255, 0.1);
        animation: idle-pulse 4s infinite ease-in-out;
    }

    .orb-core.processing {
        background: radial-gradient(circle at 30% 30%, rgba(168, 85, 247, 0.8), rgba(126, 34, 206, 1) 50%, rgba(60, 10, 80, 1) 100%);
        animation: processing-spin 2s infinite linear;
    }

    @keyframes idle-pulse { 0%, 100% { opacity: 0.85; } 50% { opacity: 1; } }
    @keyframes processing-spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }

    .chat-message { padding: 1rem; border-radius: 10px; margin-bottom: 0.8rem; }
    .user-message { background-color: rgba(56, 189, 248, 0.1); border-left: 3px solid #38bdf8; text-align: right; }
    .bot-message { background-color: rgba(31, 41, 55, 0.5); border-left: 3px solid rgba(168, 85, 247, 0.5); }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# 🧠 FUNÇÕES DE BACKEND
# ═══════════════════════════════════════════════════════════

@st.cache_resource
def get_ai_chain():
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.3, max_tokens=150)
    prompt = ChatPromptTemplate.from_template(
        "Você é o MonkeyAI. Responda de forma breve e direta.\nHuman: {input}\nMonkeyAI:"
    )
    return prompt | llm | StrOutputParser()

def transcribe_audio(audio_bytes):
    """
    CORREÇÃO PRINCIPAL: Lê o bytesIO como um arquivo WAV
    """
    r = sr.Recognizer()
    try:
        # Usa AudioFile em vez de AudioData porque o recorder entrega um WAV com header
        with io.BytesIO(audio_bytes) as audio_file:
            with sr.AudioFile(audio_file) as source:
                audio_data = r.record(source)
                text = r.recognize_google(audio_data, language='pt-BR')
                return text
    except sr.UnknownValueError:
        return None
    except Exception as e:
        st.error(f"Erro ao transcrever: {e}")
        return None

def text_to_speech_gtts(text, lang_choice):
    try:
        clean_text = re.sub(r'http\S+', 'link', text)
        tld = 'com.br' if lang_choice == 'Brasil' else 'pt'
        tts = gTTS(text=clean_text, lang='pt', tld=tld, slow=False)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp
    except Exception as e:
        st.warning(f"Erro ao gerar voz: {e}")
        return None

def check_commands(text):
    text_lower = text.lower()
    
    # YouTube
    if re.search(r'\b(tocar|ouvir|ver|assistir|youtube)\b', text_lower):
        term = re.sub(r'\b(tocar|ouvir|ver|assistir|video|musica|no|youtube)\b', '', text_lower, flags=re.IGNORECASE).strip()
        if term:
            return "youtube", f"https://www.youtube.com/results?search_query={term.replace(' ', '+')}", f"Abrindo YouTube: {term}"
    
    # Google
    if re.search(r'\b(pesquisar|buscar|google)\b', text_lower):
        term = re.sub(r'\b(pesquisar|buscar|google|sobre)\b', '', text_lower, flags=re.IGNORECASE).strip()
        if term:
            return "google", f"https://www.google.com/search?q={term.replace(' ', '+')}", f"Pesquisando: {term}"
    
    return "chat", None, None

def open_link_js(url):
    components.html(f"<script>window.open('{url}', '_blank');</script>", height=0)

# ═══════════════════════════════════════════════════════════
# 📱 INTERFACE PRINCIPAL
# ═══════════════════════════════════════════════════════════
def main():
    if "messages" not in st.session_state: st.session_state.messages = []

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Config")
        input_mode = st.radio("Modo:", ("🗣️ Voz", "⌨️ Texto"))
        voice_accent = st.selectbox("Sotaque:", ("Brasil", "Portugal")) if "Voz" in input_mode else "Brasil"
        if st.button("🧹 Limpar"):
            st.session_state.messages = []
            st.rerun()

    # Título
    st.markdown("<h1 style='text-align: center; margin-bottom: 80px;'>MONKEY<span style='color:#38bdf8;'>AI</span></h1>", unsafe_allow_html=True)

    # Variável para processamento
    final_input = None

    # --- ÁREA DO ORBE (MODO VOZ) ---
    if "Voz" in input_mode:
        # Visual do Orbe (Fixo no fundo)
        orb_class = "processing" if "audio_bytes" in locals() and audio_bytes else ""
        
        st.markdown(f"""
        <div class="orb-container">
            <div class="orb-core {orb_class}"></div>
            <div style="margin-top: 20px; color: #aaa; font-size: 14px;">Clique no centro para falar</div>
        </div>
        """, unsafe_allow_html=True)
        
        # O gravador real (fica por cima do orbe devido ao CSS)
        # Atenção: O audio_recorder do streamlit reseta no rerun, então processamos imediatamente
        audio_bytes = audio_recorder(
            text="",
            recording_color="#ef4444",
            neutral_color="#38bdf8",
            icon_size="2x",
            key="recorder"
        )
        
        if audio_bytes:
            # Processamento Imediato (Sem st.rerun antes de salvar)
            final_input = transcribe_audio(audio_bytes)
            if not final_input:
                st.toast("⚠️ Não entendi o áudio")

    else:
        # MODO TEXTO
        final_input = st.chat_input("Digite aqui...")

    # --- LÓGICA CENTRAL ---
    if final_input:
        # Adiciona mensagem do usuário
        st.session_state.messages.append({"role": "user", "content": final_input})
        
        # Verifica comandos
        cmd, url, reply = check_commands(final_input)
        ai_response = ""

        if cmd != "chat":
            ai_response = reply
            st.toast(f"🚀 {reply}")
            open_link_js(url)
        else:
            with st.spinner("Processando..."):
                chain = get_ai_chain()
                ai_response = chain.invoke({"input": final_input})
        
        # Salva resposta
        st.session_state.messages.append({"role": "bot", "content": ai_response})
        
        # Toca áudio se estiver no modo voz
        if "Voz" in input_mode:
            audio_fp = text_to_speech_gtts(ai_response, voice_accent)
            if audio_fp:
                # Autoplay true para resposta imediata
                st.audio(audio_fp, format='audio/mp3', autoplay=True)
        
        # Rerun APENAS se for modo texto para limpar o input
        if "Texto" in input_mode:
            st.rerun()

    # --- HISTÓRICO ---
    if st.session_state.messages:
        st.divider()
        for msg in reversed(st.session_state.messages):
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
