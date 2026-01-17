import os
import io
import re
import time
import streamlit as st
import streamlit.components.v1 as components
from gtts import gTTS
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from streamlit_mic_recorder import mic_recorder
from enum import Enum

# ═══════════════════════════════════════════════════════════
# ⚙️ CONFIGURAÇÃO INICIAL (Performance & Segurança)
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Jarvis AI",
    page_icon="⚛️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Carregar Chaves de Segurança
load_dotenv()
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

if not os.getenv("GROQ_API_KEY"):
    st.error("⚠️ ERRO: Chave API não encontrada!")
    st.stop()

# ═══════════════════════════════════════════════════════════
# 🎨 DESIGN "JARVIS" (A BOLA ANIMADA)
# ═══════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* Fundo Preto Profundo */
    .stApp {
        background-color: #000000;
        background-image: radial-gradient(circle at center, #1a1a2e 0%, #000000 100%);
        color: #00d4ff;
    }

    /* Esconde elementos padrões do Streamlit para limpar a tela */
    #MainMenu, header, footer {visibility: hidden;}
    div[data-testid="stToolbar"] {display: none;}

    /* ESTILO DA BOLA (O botão de microfone) */
    .stButton button, div[data-testid="stVerticalBlock"] button {
        background: radial-gradient(circle, #00d4ff 0%, #005f73 100%) !important;
        border: 2px solid #00d4ff !important;
        border-radius: 50% !important;
        width: 180px !important;
        height: 180px !important;
        font-size: 60px !important;
        margin: 0 auto !important;
        display: block !important;
        box-shadow: 0 0 30px #00d4ff, inset 0 0 20px #ffffff !important;
        transition: all 0.3s ease !important;
        color: white !important;
    }

    /* Animação ao passar o mouse ou clicar */
    .stButton button:hover, div[data-testid="stVerticalBlock"] button:active {
        transform: scale(1.1);
        box-shadow: 0 0 60px #00d4ff, inset 0 0 30px #ffffff !important;
    }

    /* Animação de "Respirando" para a bola */
    @keyframes pulse {
        0% { box-shadow: 0 0 30px #00d4ff; }
        50% { box-shadow: 0 0 60px #00d4ff, 0 0 100px #00d4ff; }
        100% { box-shadow: 0 0 30px #00d4ff; }
    }
    
    /* Container centralizado */
    .mic-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 60vh;
        animation: pulse 3s infinite;
        border-radius: 50%;
    }

    /* Balões de Texto Minimalistas */
    .chat-msg {
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        font-family: sans-serif;
        font-size: 18px;
    }
    .user { text-align: right; color: #aaa; font-style: italic; }
    .bot { text-align: center; color: #00d4ff; font-weight: bold; text-shadow: 0 0 10px rgba(0,212,255,0.5); }

</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# 🧠 CÉREBRO OTIMIZADO (Cache + IA Rápida)
# ═══════════════════════════════════════════════════════════

@st.cache_resource
def get_llm():
    """Carrega a IA na memória para não travar a cada mensagem"""
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant", 
        temperature=0.3, # Mais preciso e rápido
        max_tokens=100   # Respostas curtas para ser ágil
    )
    prompt = ChatPromptTemplate.from_template(
        "Você é o Jarvis. Responda em 1 frase curta e direta em PT-BR.\nHistórico: {history}\nHumano: {input}"
    )
    return prompt | llm | StrOutputParser()

def process_audio_command(text):
    """Processa comandos de redirecionamento instantâneo"""
    text_lower = text.lower()
    
    # Comandos Rápidos (Regex)
    if re.search(r'\b(tocar|ouvir|ver|assistir)\b', text_lower):
        term = re.sub(r'\b(tocar|ouvir|ver|assistir|video|musica|no|youtube)\b', '', text_lower, flags=re.IGNORECASE).strip()
        return "youtube", term, f"https://www.youtube.com/results?search_query={term}"
    
    if re.search(r'\b(pesquisar|buscar|google)\b', text_lower):
        term = re.sub(r'\b(pesquisar|buscar|google|sobre|o que é)\b', '', text_lower, flags=re.IGNORECASE).strip()
        return "google", term, f"https://www.google.com/search?q={term}"
        
    return "chat", text, None

def open_link(url):
    """JavaScript para abrir link forçado"""
    js = f"<script>window.open('{url}', '_blank').focus();</script>"
    components.html(js, height=0)

# ═══════════════════════════════════════════════════════════
# 📱 INTERFACE PRINCIPAL
# ═══════════════════════════════════════════════════════════

def main():
    # Inicializa sessão
    if "messages" not in st.session_state: st.session_state.messages = []
    
    # Título Flutuante
    st.markdown("<h1 style='text-align: center; color: white; opacity: 0.8;'>JARVIS <span style='font-size: 15px; color: #00d4ff;'>V3.0</span></h1>", unsafe_allow_html=True)

    # --- A BOLA (Microfone) ---
    # Colocamos o gravador dentro de colunas para centralizar perfeitamente
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="mic-container">', unsafe_allow_html=True)
        # O componente de áudio que vira a bola
        audio_data = mic_recorder(
            start_prompt="🎙️", 
            stop_prompt="✋", 
            key='recorder', 
            format="wav",
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # --- LÓGICA DE PROCESSAMENTO ---
    if audio_data:
        # 1. Transcrever Áudio (Sem usar speech_recognition pesado)
        # Precisamos de uma transcrição. Como o mic_recorder só entrega bytes, 
        # usamos uma função leve ou API. Para manter simples e grátis, usamos GoogleSpeech aqui:
        import speech_recognition as sr
        r = sr.Recognizer()
        
        try:
            text = ""
            with io.BytesIO(audio_data['bytes']) as source_audio:
                with sr.AudioFile(source_audio) as source:
                    audio_content = r.record(source)
                    text = r.recognize_google(audio_content, language='pt-BR')
            
            if text:
                # Mostra o que você falou
                st.markdown(f"<div class='chat-msg user'>🗣️ {text}</div>", unsafe_allow_html=True)
                
                # Processa comando
                type, term, url = process_audio_command(text)
                
                response_text = ""
                
                if url:
                    response_text = f"Abrindo {type}: {term}..."
                    st.toast(f"🚀 {response_text}")
                    # Abertura automática
                    open_link(url)
                else:
                    # IA Rápida
                    chain = get_llm()
                    # Histórico simples (últimas 2 falas para ser rápido)
                    hist_str = "\n".join([m for m in st.session_state.messages[-2:]]) 
                    response_text = chain.invoke({"history": hist_str, "input": text})
                    st.session_state.messages.append(f"Eu: {text}")
                    st.session_state.messages.append(f"Jarvis: {response_text}")

                # Resposta Visual
                st.markdown(f"<div class='chat-msg bot'>{response_text}</div>", unsafe_allow_html=True)
                
                # Áudio de Resposta (TTS)
                try:
                    tts = gTTS(text=response_text, lang='pt')
                    audio_fp = io.BytesIO()
                    tts.write_to_fp(audio_fp)
                    st.audio(audio_fp, format='audio/mp3', autoplay=True)
                except:
                    pass
                
        except Exception as e:
            st.error("Não entendi...")

    # Instrução visual sutil
    st.markdown("<p style='text-align: center; color: #555; margin-top: 50px;'>Toque no núcleo para falar</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
