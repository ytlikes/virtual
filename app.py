import os
import time
import io
import streamlit as st
from gtts import gTTS
import speech_recognition as sr
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from streamlit_mic_recorder import mic_recorder

# 🔧 CONFIGURAÇÕES
load_dotenv()

# Verificação de segurança da API Key
if not os.getenv("GROQ_API_KEY"):
    st.error("❌ ERRO: Chave da API Groq não encontrada. Configure o arquivo .env")
    st.stop()

Config = {
    "MODEL_NAME": "llama-3.1-8b-instant",
    "PAGE_TITLE": "Jarvis Mobile",
    "PAGE_ICON": "📱",
}

# 🎨 CSS OTIMIZADO PARA MOBILE
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    /* Botões grandes para toque */
    .stButton > button {
        width: 100%;
        height: 60px;
        border-radius: 12px;
        font-size: 20px;
        font-weight: bold;
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        border: none;
        color: white;
    }
    /* Estilo do chat */
    .chat-bubble {
        padding: 15px;
        border-radius: 15px;
        margin-bottom: 10px;
        font-size: 16px;
    }
    .user-bubble {
        background-color: #2b313e;
        border-bottom-right-radius: 2px;
        text-align: right;
    }
    .bot-bubble {
        background-color: #0052cc;
        border-bottom-left-radius: 2px;
    }
</style>
""", unsafe_allow_html=True)

# 🧠 LÓGICA DA IA
def get_ai_response(text):
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="history")
    
    template = """
    Você é um assistente mobile útil e breve. 
    Responda de forma concisa pois o usuário está no celular.
    Histórico: {history}
    Humano: {input}
    IA:"""
    
    prompt = PromptTemplate(input_variables=["history", "input"], template=template)
    llm = ChatGroq(model_name=Config["MODEL_NAME"], temperature=0.6)
    chain = LLMChain(llm=llm, prompt=prompt, memory=st.session_state.memory)
    
    return chain.run(input=text)

# 🔊 FUNÇÃO DE FALA (TEXT-TO-SPEECH)
def text_to_speech(text):
    try:
        # Cria um arquivo MP3 na memória
        tts = gTTS(text=text, lang='pt', slow=False)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp
    except Exception as e:
        st.error(f"Erro no áudio: {e}")
        return None

# 🎤 PROCESSAR ÁUDIO DO MICROFONE
def process_audio(audio_bytes):
    r = sr.Recognizer()
    try:
        # Converte bytes para arquivo de áudio legível
        audio_file = io.BytesIO(audio_bytes)
        with sr.AudioFile(audio_file) as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data, language='pt-BR')
            return text
    except sr.UnknownValueError:
        return None
    except Exception as e:
        st.error(f"Erro ao processar áudio: {e}")
        return None

# 📱 INTERFACE PRINCIPAL
def main():
    st.title("📱 Jarvis Mobile")
    
    # Inicializar chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- ÁREA DE INPUT ---
    st.write("### 🎙️ Fale ou Digite")
    
    # 1. Botão de Gravação (Componente especial para Web/Mobile)
    col1, col2 = st.columns([1, 4])
    with col1:
        # Este componente grava direto no navegador do celular
        audio_data = mic_recorder(
            start_prompt="🎤",
            stop_prompt="🛑",
            key='recorder',
            format="wav"
        )
    
    with col2:
        text_input = st.text_input("Digite...", key="text_input", label_visibility="collapsed")

    # Lógica de processamento
    user_text = None

    # Se gravou áudio
    if audio_data:
        with st.spinner("Ouvindo..."):
            user_text = process_audio(audio_data['bytes'])
    
    # Se digitou texto (prioridade se ambos existirem)
    if text_input:
        user_text = text_input

    # --- PROCESSAMENTO E RESPOSTA ---
    if user_text:
        # Adiciona mensagem do usuário
        st.session_state.messages.append({"role": "user", "content": user_text})
        
        # Gera resposta da IA
        with st.spinner("Pensando..."):
            ai_response = get_ai_response(user_text)
            st.session_state.messages.append({"role": "bot", "content": ai_response})
            
            # Gera áudio da resposta
            audio_response = text_to_speech(ai_response)
            if audio_response:
                st.audio(audio_response, format='audio/mp3', autoplay=True)

    # --- EXIBIÇÃO DO CHAT (Invertido: Mais recente no topo) ---
    st.write("---")
    for msg in reversed(st.session_state.messages):
        role_class = "user-bubble" if msg["role"] == "user" else "bot-bubble"
        icon = "👤" if msg["role"] == "user" else "🤖"
        st.markdown(f"""
        <div class="chat-bubble {role_class}">
            <b>{icon}</b> {msg['content']}
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()