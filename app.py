import os
import io
import re
import time
import streamlit as st
import streamlit.components.v1 as components
from gtts import gTTS
import speech_recognition as sr
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from streamlit_mic_recorder import mic_recorder

# ═══════════════════════════════════════════════════════════
# ⚙️ CONFIGURAÇÃO
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Jarvis V5",
    page_icon="⚛️",
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
# 🎨 CSS CORRIGIDO (ORBE GARANTIDO)
# ═══════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* Fundo Preto */
    .stApp {
        background-color: #000000;
        background-image: radial-gradient(circle at center, #111827 0%, #000000 100%);
        color: #e0f2fe;
    }
    
    #MainMenu, footer {visibility: hidden;}

    /* --- O ORBE (CSS Focado no Iframe) --- */
    /* Cria uma área brilhante atrás do gravador */
    .orb-wrapper {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 50px;
        margin-bottom: 30px;
        height: 200px; /* Garante altura para não sumir */
    }

    /* Estiliza o componente de gravação (iframe) */
    .orb-wrapper iframe {
        border-radius: 50% !important;
        width: 150px !important;
        height: 150px !important;
        
        /* O Brilho Azul (Orbe) */
        box-shadow: 
            0 0 30px rgba(56, 189, 248, 0.6),
            0 0 60px rgba(3, 105, 161, 0.4),
            inset 0 0 20px rgba(255, 255, 255, 0.2);
            
        background: radial-gradient(circle, #0c4a6e 0%, #000000 100%);
        border: 2px solid #38bdf8;
        animation: pulse 3s infinite ease-in-out;
    }

    @keyframes pulse {
        0% { transform: scale(1); box-shadow: 0 0 30px rgba(56, 189, 248, 0.6); }
        50% { transform: scale(1.05); box-shadow: 0 0 80px rgba(56, 189, 248, 0.8); }
        100% { transform: scale(1); box-shadow: 0 0 30px rgba(56, 189, 248, 0.6); }
    }

    /* Estilo do Chat */
    .user-message {
        background: rgba(56, 189, 248, 0.1);
        border-right: 3px solid #38bdf8;
        padding: 10px;
        border-radius: 10px;
        text-align: right;
        margin-bottom: 10px;
    }
    .bot-message {
        background: rgba(255, 255, 255, 0.05);
        border-left: 3px solid #ccc;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# 🧠 FUNÇÕES
# ═══════════════════════════════════════════════════════════

@st.cache_resource
def get_ai_chain():
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.4, max_tokens=150)
    prompt = ChatPromptTemplate.from_template(
        "Você é o Jarvis. Responda curto e direto.\nHistórico: {history}\nHumano: {input}\nJarvis:"
    )
    return prompt | llm | StrOutputParser()

def transcribe_audio(audio_bytes):
    r = sr.Recognizer()
    try:
        with io.BytesIO(audio_bytes) as source_audio:
            with sr.AudioFile(source_audio) as source:
                audio_content = r.record(source)
                return r.recognize_google(audio_content, language='pt-BR')
    except: return None

def text_to_speech_gtts(text):
    try:
        clean_text = re.sub(r'http\S+', '', text)
        tts = gTTS(text=clean_text, lang='pt', slow=False)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp
    except: return None

def check_commands(text):
    text_lower = text.lower()
    if re.search(r'\b(tocar|ouvir|ver|youtube)\b', text_lower):
        term = re.sub(r'\b(tocar|ouvir|ver|video|musica|no|youtube)\b', '', text_lower, flags=re.IGNORECASE).strip()
        return "youtube", f"https://www.youtube.com/results?search_query={term}", f"Abrindo YouTube: {term}"
    if re.search(r'\b(pesquisar|buscar|google)\b', text_lower):
        term = re.sub(r'\b(pesquisar|buscar|google|sobre)\b', '', text_lower, flags=re.IGNORECASE).strip()
        return "google", f"https://www.google.com/search?q={term}", f"Pesquisando: {term}"
    return "chat", None, None

def open_link_js(url):
    js = f"<script>window.open('{url}', '_blank').focus();</script>"
    components.html(js, height=0)

# ═══════════════════════════════════════════════════════════
# 📱 APP PRINCIPAL
# ═══════════════════════════════════════════════════════════
def main():
    if "messages" not in st.session_state: st.session_state.messages = []

    # Sidebar para Modos
    with st.sidebar:
        mode = st.radio("Modo:", ["Voz (Orbe)", "Texto"])
        if st.button("Limpar Conversa"):
            st.session_state.messages = []
            st.rerun()

    st.markdown("<h1 style='text-align: center;'>JARVIS <span style='color:#38bdf8'>V5</span></h1>", unsafe_allow_html=True)

    user_input = None

    # --- ÁREA DO ORBE ---
    if mode == "Voz (Orbe)":
        # Container que força o CSS a funcionar
        st.markdown('<div class="orb-wrapper">', unsafe_allow_html=True)
        
        # O Gravador (Aparecerá como botão dentro do brilho)
        audio_data = mic_recorder(
            start_prompt="🎙️", # Ícone visível
            stop_prompt="⬛", 
            key='orb',
            format="wav",
            use_container_width=False 
        )
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #555;'>Clique no ícone para falar</p>", unsafe_allow_html=True)
        
        if audio_data:
            user_input = transcribe_audio(audio_data['bytes'])
    else:
        user_input = st.chat_input("Digite...")

    # --- PROCESSAMENTO ---
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        cmd, url, reply = check_commands(user_input)
        
        final_reply = ""
        
        if cmd != "chat":
            final_reply = reply
            st.toast(f"🚀 {reply}")
            open_link_js(url)
        else:
            with st.spinner("..."):
                chain = get_ai_chain()
                hist_text = "\n".join([m['content'] for m in st.session_state.messages[-3:]])
                final_reply = chain.invoke({"history": hist_text, "input": user_input})

        st.session_state.messages.append({"role": "bot", "content": final_reply})
        
        if mode == "Voz (Orbe)":
            audio = text_to_speech_gtts(final_reply)
            if audio: st.audio(audio, format='audio/mp3', autoplay=True)
        
        if mode == "Texto": st.rerun()

    # --- HISTÓRICO ---
    st.divider()
    for msg in reversed(st.session_state.messages):
        cls = "user-message" if msg['role'] == "user" else "bot-message"
        st.markdown(f"<div class='{cls}'>{msg['content']}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
