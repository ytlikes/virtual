import os
import io
import re
import time
import logging
import streamlit as st
import streamlit.components.v1 as components
from gtts import gTTS
import speech_recognition as sr
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from streamlit_mic_recorder import mic_recorder
from enum import Enum
from typing import Optional, Tuple

# ═══════════════════════════════════════════════════════════
# 🔧 CONFIGURAÇÕES E SEGURANÇA
# ═══════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Jarvis Mobile",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Carregamento de Chaves
load_dotenv()
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

if not os.getenv("GROQ_API_KEY"):
    st.error("❌ ERRO: Chave da API Groq não encontrada! Configure no .env ou Secrets.")
    st.stop()

class Config:
    MODEL_NAME = "llama-3.1-8b-instant"
    LANGUAGE = 'pt-BR'
    AI_TEMPLATE = """Você é um assistente pessoal inteligente.
    Diretrizes:
    - Seja extremamente breve (1 frase curta).
    - Aja naturalmente.
    
    Histórico: {historico_conversa}
    Usuário: {input}
    Assistente:"""

class CommandType(Enum):
    YOUTUBE = "youtube"
    GOOGLE = "google"
    CHAT = "chat"

# ═══════════════════════════════════════════════════════════
# 🎨 ESTILOS CSS (FUTURISTA/NEON)
# ═══════════════════════════════════════════════════════════
st.markdown("""
<style>
    .stApp {
        background-color: #000000;
        background-image: radial-gradient(circle at 50% 50%, #1e1e2f 0%, #000000 100%);
        color: #00ffcc;
    }
    
    /* Balões de Chat */
    .chat-bubble {
        padding: 12px 18px;
        border-radius: 20px;
        margin-bottom: 10px;
        font-family: 'Segoe UI', sans-serif;
        box-shadow: 0 0 10px rgba(0, 255, 204, 0.1);
    }
    
    .user-bubble {
        background: rgba(0, 255, 204, 0.1);
        border: 1px solid #00ffcc;
        color: #00ffcc;
        text-align: right;
        margin-left: 20%;
        border-bottom-right-radius: 4px;
    }
    
    .bot-bubble {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: #ffffff;
        margin-right: 20%;
        border-bottom-left-radius: 4px;
    }

    /* Botão de Gravar */
    div[data-testid="stVerticalBlock"] button {
        background-color: #ff0055 !important;
        border: none;
        border-radius: 50%;
        width: 70px;
        height: 70px;
        box-shadow: 0 0 20px #ff0055;
        transition: transform 0.2s;
    }
    div[data-testid="stVerticalBlock"] button:active {
        transform: scale(0.9);
        box-shadow: 0 0 10px #ff0055;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# 🛠️ FERRAMENTAS (JAVASCRIPT E ÁUDIO)
# ═══════════════════════════════════════════════════════════

def open_url_automatically(url):
    """Injeta JavaScript para abrir aba automaticamente"""
    js = f"""
    <script>
        window.open('{url}', '_blank').focus();
    </script>
    """
    components.html(js, height=0, width=0)

class AudioManager:
    @staticmethod
    def process_mic_input(audio_bytes: bytes) -> Optional[str]:
        r = sr.Recognizer()
        try:
            audio_file = io.BytesIO(audio_bytes)
            with sr.AudioFile(audio_file) as source:
                audio_data = r.record(source)
                return r.recognize_google(audio_data, language=Config.LANGUAGE)
        except:
            return None

    @staticmethod
    def text_to_speech(text: str):
        try:
            clean_text = re.sub(r'http\S+', '', text) # Não ler URLs
            tts = gTTS(text=clean_text, lang='pt', slow=False)
            audio_fp = io.BytesIO()
            tts.write_to_fp(audio_fp)
            audio_fp.seek(0)
            return audio_fp
        except:
            return None

class CommandProcessor:
    YOUTUBE_TRIGGERS = [r'\btocar\b', r'\bouvir\b', r'\bmusica\b', r'\bver\b.*\bvideo\b', r'\byoutube\b']
    GOOGLE_TRIGGERS = [r'\bpesquisar\b', r'\bbuscar\b', r'\bgoogle\b']
    
    @classmethod
    def process(cls, text: str) -> Tuple[CommandType, Optional[str]]:
        text_lower = text.lower().strip()
        for pattern in cls.YOUTUBE_TRIGGERS:
            if re.search(pattern, text_lower):
                return CommandType.YOUTUBE, cls._extract_term(text_lower, cls.YOUTUBE_TRIGGERS)
        for pattern in cls.GOOGLE_TRIGGERS:
            if re.search(pattern, text_lower):
                return CommandType.GOOGLE, cls._extract_term(text_lower, cls.GOOGLE_TRIGGERS)
        return CommandType.CHAT, None

    @staticmethod
    def _extract_term(text: str, patterns: list) -> str:
        result = text
        for pattern in patterns:
            result = re.sub(pattern, '', result, flags=re.IGNORECASE)
        words = [w for w in result.split() if w not in ['a', 'o', 'de', 'para', 'em', 'video', 'musica']]
        return ' '.join(words).strip()

class AIManager:
    def __init__(self):
        if 'chain' not in st.session_state:
            prompt = PromptTemplate(input_variables=["historico_conversa", "input"], template=Config.AI_TEMPLATE)
            llm = ChatGroq(model_name=Config.MODEL_NAME, temperature=0.6)
            memory = ConversationBufferMemory(memory_key="historico_conversa", input_key="input")
            st.session_state.chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

    def get_response(self, text: str) -> str:
        try:
            return st.session_state.chain.run(input=text)
        except:
            return "Estou reiniciando meus sistemas..."

# ═══════════════════════════════════════════════════════════
# 📱 APP PRINCIPAL
# ═══════════════════════════════════════════════════════════

def main():
    if 'messages' not in st.session_state: st.session_state.messages = []
    
    # Header Futurista
    st.markdown("<h1 style='text-align: center; text-shadow: 0 0 10px #00ffcc;'>JARVIS <span style='font-size: 15px; vertical-align: top;'>ONLINE</span></h1>", unsafe_allow_html=True)

    # --- INPUT ---
    col_mic, col_text = st.columns([1, 4])
    with col_mic:
        audio_data = mic_recorder(start_prompt="🔊", stop_prompt="⏹️", key='recorder', format="wav", use_container_width=True)
    with col_text:
        text_input = st.chat_input("Comando de texto...")

    user_input = AudioManager.process_mic_input(audio_data['bytes']) if audio_data else text_input

    # --- PROCESSAMENTO ---
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        cmd_type, term = CommandProcessor.process(user_input)
        
        bot_reply = ""
        redirect_url = None

        if cmd_type == CommandType.YOUTUBE:
            redirect_url = f"https://www.youtube.com/results?search_query={term}"
            bot_reply = f"Executando protocolo de vídeo: {term}"
            st.toast(f"🚀 Redirecionando para YouTube: {term}", icon="🎵")
            
        elif cmd_type == CommandType.GOOGLE:
            redirect_url = f"https://www.google.com/search?q={term}"
            bot_reply = f"Pesquisando na rede: {term}"
            st.toast(f"🚀 Pesquisando no Google: {term}", icon="🔍")
            
        else:
            bot_reply = AIManager().get_response(user_input)

        # Salva e Fala
        st.session_state.messages.append({"role": "bot", "content": bot_reply})
        audio = AudioManager.text_to_speech(bot_reply)
        if audio: st.audio(audio, format="audio/mp3", autoplay=True)

        # REDIRECIONAMENTO AUTOMÁTICO (JavaScript)
        if redirect_url:
            open_url_automatically(redirect_url)
            # Link de backup caso o pop-up bloqueie
            st.markdown(f"<br><a href='{redirect_url}' target='_blank' style='background: #333; color: white; padding: 10px; border-radius: 10px; text-decoration: none;'>🔗 Clique aqui se não abrir automaticamente</a>", unsafe_allow_html=True)
            time.sleep(2) # Pausa dramática antes do rerun
            st.rerun()
            
        # Rerun para limpar input de texto (se for chat normal)
        if cmd_type == CommandType.CHAT:
            time.sleep(1)
            st.rerun()

    # --- HISTÓRICO ---
    for msg in reversed(st.session_state.messages):
        role_class = "user-bubble" if msg["role"] == "user" else "bot-bubble"
        st.markdown(f"<div class='chat-bubble {role_class}'>{msg['content']}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
