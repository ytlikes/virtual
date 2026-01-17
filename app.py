import os
import io
import re
import time
import logging
import streamlit as st
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

# Configuração da Página deve ser a primeira linha do Streamlit
st.set_page_config(
    page_title="Jarvis Mobile",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Carregamento de Chaves (Híbrido: Local .env ou Nuvem Secrets)
load_dotenv()

if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

if not os.getenv("GROQ_API_KEY"):
    st.error("❌ ERRO: Chave da API Groq não encontrada! Configure no .env ou Secrets.")
    st.stop()

class Config:
    MODEL_NAME = "llama-3.1-8b-instant"
    LANGUAGE = 'pt-BR'
    AI_TEMPLATE = """Você é um assistente pessoal inteligente, gentil e eficiente.
    Seu estilo é profissional mas amigável, como uma versão moderna do Jarvis.
    
    Diretrizes:
    - Seja conciso (máximo 2 frases), pois o usuário está no celular.
    - Responda sempre em português brasileiro.
    - Use emojis com moderação.
    
    Histórico: {historico_conversa}
    Usuário: {input}
    Assistente:"""

class CommandType(Enum):
    YOUTUBE = "youtube"
    GOOGLE = "google"
    CHAT = "chat"

# ═══════════════════════════════════════════════════════════
# 🎨 ESTILOS CSS (FUTURISTA/MOBILE)
# ═══════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* Fundo e Geral */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f1f5f9;
    }
    
    /* Balões de Chat */
    .chat-bubble {
        padding: 15px;
        border-radius: 15px;
        margin-bottom: 12px;
        font-size: 16px;
        line-height: 1.5;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    
    .user-bubble {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: white;
        border-bottom-right-radius: 2px;
        margin-left: 15%;
        text-align: right;
    }
    
    .bot-bubble {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #e2e8f0;
        border-bottom-left-radius: 2px;
        margin-right: 15%;
    }

    /* Links de Comandos */
    .command-link {
        display: inline-block;
        margin-top: 10px;
        padding: 8px 16px;
        background-color: #ef4444; /* Vermelho YouTube */
        color: white !important;
        text-decoration: none;
        border-radius: 20px;
        font-weight: bold;
    }
    .google-link {
        background-color: #3b82f6; /* Azul Google */
    }

    /* Botão de Gravar */
    div[data-testid="stVerticalBlock"] button {
        border-radius: 50%;
        width: 60px;
        height: 60px;
    }
    
    /* Título */
    h1 {
        background: linear-gradient(90deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# 🧠 LÓGICA (ÁUDIO, IA, COMANDOS)
# ═══════════════════════════════════════════════════════════

class AudioManager:
    """Gerencia áudio adaptado para Web/Mobile"""
    
    @staticmethod
    def process_mic_input(audio_bytes: bytes) -> Optional[str]:
        """Converte áudio do navegador em texto"""
        r = sr.Recognizer()
        try:
            audio_file = io.BytesIO(audio_bytes)
            with sr.AudioFile(audio_file) as source:
                audio_data = r.record(source)
                text = r.recognize_google(audio_data, language=Config.LANGUAGE)
                return text
        except Exception as e:
            return None

    @staticmethod
    def text_to_speech(text: str):
        """Gera áudio MP3 para o navegador tocar"""
        try:
            # Remove links/markdown para não ler códigos estranhos
            clean_text = re.sub(r'\[.*?\]\(.*?\)', '', text) 
            tts = gTTS(text=clean_text, lang='pt', slow=False)
            audio_fp = io.BytesIO()
            tts.write_to_fp(audio_fp)
            audio_fp.seek(0)
            return audio_fp
        except Exception:
            return None

class CommandProcessor:
    """Processa comandos (Adaptado para Links Web)"""
    
    YOUTUBE_TRIGGERS = [r'\btocar\b', r'\bouvir\b', r'\bmusica\b', r'\bver\b.*\bvideo\b', r'\byoutube\b']
    GOOGLE_TRIGGERS = [r'\bpesquisar\b', r'\bbuscar\b', r'\bgoogle\b']
    
    @classmethod
    def process(cls, text: str) -> Tuple[CommandType, Optional[str]]:
        text_lower = text.lower().strip()
        
        for pattern in cls.YOUTUBE_TRIGGERS:
            if re.search(pattern, text_lower):
                term = cls._extract_term(text_lower, cls.YOUTUBE_TRIGGERS)
                return CommandType.YOUTUBE, term
                
        for pattern in cls.GOOGLE_TRIGGERS:
            if re.search(pattern, text_lower):
                term = cls._extract_term(text_lower, cls.GOOGLE_TRIGGERS)
                return CommandType.GOOGLE, term
                
        return CommandType.CHAT, None

    @staticmethod
    def _extract_term(text: str, patterns: list) -> str:
        result = text
        for pattern in patterns:
            result = re.sub(pattern, '', result, flags=re.IGNORECASE)
        stop_words = ['a', 'o', 'de', 'no', 'na', 'para', 'em', 'video', 'musica', 'sobre']
        words = [w for w in result.split() if w not in stop_words]
        return ' '.join(words).strip()

class AIManager:
    """Gerencia LangChain"""
    def __init__(self):
        if 'chain' not in st.session_state:
            prompt = PromptTemplate(
                input_variables=["historico_conversa", "input"],
                template=Config.AI_TEMPLATE
            )
            llm = ChatGroq(model_name=Config.MODEL_NAME, temperature=0.6)
            memory = ConversationBufferMemory(memory_key="historico_conversa", input_key="input")
            st.session_state.chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

    def get_response(self, text: str) -> str:
        try:
            return st.session_state.chain.run(input=text)
        except:
            return "Desculpe, estou em manutenção."

# ═══════════════════════════════════════════════════════════
# 📱 INTERFACE PRINCIPAL
# ═══════════════════════════════════════════════════════════

def main():
    st.markdown("<h1>🤖 Jarvis Mobile</h1>", unsafe_allow_html=True)
    
    # Inicialização
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    ai_manager = AIManager()

    # --- ÁREA DE INPUT ---
    # Container fixo ou no topo para facilitar no mobile
    col_mic, col_text = st.columns([1, 4])
    
    with col_mic:
        audio_data = mic_recorder(
            start_prompt="🎤",
            stop_prompt="⏹️",
            key='recorder',
            format="wav",
            use_container_width=True
        )
    
    with col_text:
        text_input = st.chat_input("Digite ou fale...")

    # Processamento
    user_input = None
    
    # 1. Checa Áudio
    if audio_data:
        user_input = AudioManager.process_mic_input(audio_data['bytes'])
    
    # 2. Checa Texto (Chat Input do Streamlit)
    if text_input:
        user_input = text_input

    # 3. Executa Lógica
    if user_input:
        # Salva msg usuário
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Processa comando ou IA
        cmd_type, term = CommandProcessor.process(user_input)
        
        bot_response = ""
        audio_response = None
        
        if cmd_type == CommandType.YOUTUBE:
            url = f"https://www.youtube.com/results?search_query={term}"
            bot_response = f"🎵 Abrindo YouTube para: **{term}**\n\n<a href='{url}' target='_blank' class='command-link'>▶️ Clique para Assistir</a>"
            audio_response = AudioManager.text_to_speech(f"Abrindo YouTube para {term}")
            
        elif cmd_type == CommandType.GOOGLE:
            url = f"https://www.google.com/search?q={term}"
            bot_response = f"🔍 Pesquisando: **{term}**\n\n<a href='{url}' target='_blank' class='command-link google-link'>🌐 Clique para Ver</a>"
            audio_response = AudioManager.text_to_speech(f"Pesquisando no Google sobre {term}")
            
        else:
            with st.spinner("Pensando..."):
                bot_response = ai_manager.get_response(user_input)
                audio_response = AudioManager.text_to_speech(bot_response)

        # Salva resposta
        st.session_state.messages.append({"role": "bot", "content": bot_response})
        
        # Toca Áudio (Autoplay)
        if audio_response:
            st.audio(audio_response, format="audio/mp3", autoplay=True)
            
        # Força atualização para limpar inputs
        st.rerun()

    # --- EXIBIÇÃO DO HISTÓRICO ---
    st.write("---")
    for msg in reversed(st.session_state.messages):
        role_class = "user-bubble" if msg["role"] == "user" else "bot-bubble"
        icon = "👤" if msg["role"] == "user" else "🤖"
        
        st.markdown(f"""
        <div class="chat-bubble {role_class}">
            <div style="font-size: 12px; opacity: 0.7; margin-bottom: 4px;">{icon} {msg['role'].upper()}</div>
            {msg['content']}
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
