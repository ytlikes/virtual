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

# ═══════════════════════════════════════════════════════════
# ⚙️ CONFIGURAÇÃO & SEGURANÇA
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MonkeyAI",
    page_icon="🐵",
    layout="centered",
    initial_sidebar_state="expanded"
)

load_dotenv()
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

if not os.getenv("GROQ_API_KEY"):
    st.error("⚠️ Erro: Chave API Groq não configurada.")
    st.stop()

# ═══════════════════════════════════════════════════════════
# 🎨 CSS APRIMORADO COM FEEDBACK VISUAL CLARO
# ═══════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* Fundo e Geral */
    .stApp {
        background-color: #000000;
        background-image: radial-gradient(circle at center, #111827 0%, #000000 100%);
        color: #e0f2fe;
    }
    
    /* Esconde elementos padrão do Streamlit */
    #MainMenu, footer {visibility: hidden;}
    .stDeployButton {display:none;}

    /* --- CONTAINER DO ORBE --- */
    .orb-section {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        min-height: 60vh;
        padding: 2rem;
    }

    /* --- STATUS VISUAL GRANDE --- */
    .status-banner {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.2), rgba(3, 105, 161, 0.1));
        border: 2px solid rgba(56, 189, 248, 0.4);
        border-radius: 16px;
        padding: 1.5rem 3rem;
        margin-bottom: 3rem;
        text-align: center;
        min-width: 400px;
        box-shadow: 0 4px 20px rgba(56, 189, 248, 0.2);
    }

    .status-banner.listening {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.3), rgba(185, 28, 28, 0.2));
        border-color: rgba(239, 68, 68, 0.6);
        box-shadow: 0 4px 20px rgba(239, 68, 68, 0.4);
        animation: pulse-border 1.5s infinite;
    }

    .status-banner.processing {
        background: linear-gradient(135deg, rgba(168, 85, 247, 0.3), rgba(126, 34, 206, 0.2));
        border-color: rgba(168, 85, 247, 0.6);
        box-shadow: 0 4px 20px rgba(168, 85, 247, 0.4);
    }

    @keyframes pulse-border {
        0%, 100% { border-color: rgba(239, 68, 68, 0.6); }
        50% { border-color: rgba(239, 68, 68, 1); }
    }

    .status-icon {
        font-size: 48px;
        margin-bottom: 0.5rem;
        display: block;
    }

    .status-text {
        font-size: 24px;
        font-weight: 600;
        color: #e0f2fe;
        margin: 0;
    }

    .status-subtext {
        font-size: 16px;
        opacity: 0.7;
        margin-top: 0.5rem;
    }

    /* --- O ORBE --- */
    .orb-wrapper {
        position: relative;
        width: 200px;
        height: 200px;
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 2rem;
    }

    /* Anéis externos */
    .orb-ring {
        position: absolute;
        border-radius: 50%;
        border: 2px solid rgba(56, 189, 248, 0.3);
        animation: ring-pulse 3s infinite ease-in-out;
    }

    .orb-ring:nth-child(1) {
        width: 220px;
        height: 220px;
    }

    .orb-ring:nth-child(2) {
        width: 260px;
        height: 260px;
        animation-delay: 0.5s;
    }

    .orb-ring:nth-child(3) {
        width: 300px;
        height: 300px;
        animation-delay: 1s;
    }

    @keyframes ring-pulse {
        0%, 100% { 
            opacity: 0.2; 
            transform: scale(0.95);
        }
        50% { 
            opacity: 0.5; 
            transform: scale(1.05);
        }
    }

    /* Núcleo do orbe */
    .orb-core {
        width: 180px;
        height: 180px;
        border-radius: 50%;
        background: radial-gradient(circle at 30% 30%, 
            rgba(56, 189, 248, 0.6), 
            rgba(3, 105, 161, 0.9) 50%, 
            rgba(0, 0, 0, 1) 100%);
        box-shadow: 
            0 0 20px rgba(56, 189, 248, 0.5),
            0 0 40px rgba(3, 105, 161, 0.3),
            0 0 60px rgba(2, 132, 199, 0.2),
            inset 0 0 30px rgba(255, 255, 255, 0.2);
        cursor: pointer;
        transition: all 0.3s ease;
        position: relative;
        z-index: 10;
        display: flex;
        justify-content: center;
        align-items: center;
        animation: idle-glow 4s infinite ease-in-out;
    }

    .orb-core:hover {
        transform: scale(1.1);
        box-shadow: 
            0 0 40px rgba(56, 189, 248, 0.8),
            0 0 60px rgba(3, 105, 161, 0.5),
            0 0 100px rgba(2, 132, 199, 0.3),
            inset 0 0 50px rgba(255, 255, 255, 0.4);
    }

    @keyframes idle-glow {
        0%, 100% { opacity: 0.9; }
        50% { opacity: 1; }
    }

    /* Estado OUVINDO */
    .orb-core.listening {
        background: radial-gradient(circle at 30% 30%, 
            rgba(239, 68, 68, 0.9), 
            rgba(185, 28, 28, 1) 50%, 
            rgba(0, 0, 0, 1) 100%);
        box-shadow: 
            0 0 50px rgba(239, 68, 68, 1),
            0 0 100px rgba(185, 28, 28, 0.7),
            0 0 150px rgba(153, 27, 27, 0.5),
            inset 0 0 50px rgba(255, 100, 100, 0.5);
        animation: listening-pulse 1s infinite ease-in-out;
    }

    @keyframes listening-pulse {
        0%, 100% { 
            transform: scale(1); 
        }
        50% { 
            transform: scale(1.08); 
        }
    }

    /* Estado PROCESSANDO */
    .orb-core.processing {
        background: radial-gradient(circle at 30% 30%, 
            rgba(168, 85, 247, 0.9), 
            rgba(126, 34, 206, 1) 50%, 
            rgba(0, 0, 0, 1) 100%);
        box-shadow: 
            0 0 50px rgba(168, 85, 247, 1),
            0 0 100px rgba(126, 34, 206, 0.7),
            0 0 150px rgba(107, 33, 168, 0.5);
        animation: processing-rotate 2s infinite linear;
    }

    @keyframes processing-rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    .orb-icon {
        font-size: 60px;
        text-shadow: 0 0 20px rgba(255, 255, 255, 0.8);
    }

    /* --- CHAT MESSAGES --- */
    .chat-message {
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        animation: fadeIn 0.3s ease;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .user-message {
        background-color: rgba(56, 189, 248, 0.15);
        border-left: 4px solid #38bdf8;
        align-items: flex-end;
    }

    .bot-message {
        background-color: rgba(31, 41, 55, 0.7);
        border-left: 4px solid rgba(168, 85, 247, 0.6);
        align-items: flex-start;
    }

    .message-text {
        font-family: 'Inter', sans-serif;
        font-size: 16px;
        line-height: 1.6;
    }

    .message-label {
        font-size: 12px;
        font-weight: 600;
        opacity: 0.8;
        margin-bottom: 6px;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# 🧠 FUNÇÕES DE BACKEND
# ═══════════════════════════════════════════════════════════

@st.cache_resource
def get_ai_chain():
    """Carrega a IA Llama-3 via Groq"""
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.4, max_tokens=150)
    prompt = ChatPromptTemplate.from_template(
        "Você é o MonkeyAI, um assistente inteligente. Responda de forma breve, inteligente e prestativa.\nHistórico recente:\n{history}\nHuman: {input}\nMonkeyAI:"
    )
    return prompt | llm | StrOutputParser()

def record_audio_with_silence_detection(duration=10, silence_threshold=500):
    """Grava áudio do microfone e para automaticamente quando detectar silêncio"""
    r = sr.Recognizer()
    r.energy_threshold = silence_threshold
    r.dynamic_energy_threshold = True
    
    try:
        with sr.Microphone() as source:
            # Ajusta para ruído ambiente
            r.adjust_for_ambient_noise(source, duration=0.8)
            
            # Grava com detecção de pausa
            audio_data = r.listen(
                source, 
                timeout=duration,
                phrase_time_limit=duration
            )
            
            return audio_data
            
    except sr.WaitTimeoutError:
        return None
    except Exception as e:
        st.error(f"❌ Erro ao gravar: {e}")
        return None

def transcribe_audio(audio_data):
    """Converte áudio em texto"""
    r = sr.Recognizer()
    try:
        text = r.recognize_google(audio_data, language='pt-BR')
        return text
    except sr.UnknownValueError:
        return None
    except sr.RequestError:
        st.error("❌ Erro de conexão com serviço de reconhecimento.")
        return None
    except Exception as e:
        st.error(f"❌ Erro ao transcrever: {e}")
        return None

def text_to_speech_gtts(text, lang_choice):
    """Gera áudio usando gTTS"""
    try:
        clean_text = re.sub(r'http\S+', 'link', text)
        
        language = 'pt'
        tld = 'com.br' if lang_choice == 'Brasil (Padrão)' else 'pt'
        
        tts = gTTS(text=clean_text, lang=language, tld=tld, slow=False)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp
    except Exception as e:
        st.warning(f"⚠️ Erro ao gerar voz: {e}")
        return None

def check_commands(text):
    """Verifica comandos de redirecionamento"""
    text_lower = text.lower()
    
    # YouTube
    if re.search(r'\b(tocar|ouvir|ver|assistir|youtube|vídeo|música)\b', text_lower):
        term = re.sub(
            r'\b(tocar|ouvir|ver|assistir|video|vídeo|musica|música|no|youtube|o|a|de|da|do)\b',
            '', 
            text_lower, 
            flags=re.IGNORECASE
        ).strip()
        
        if term:
            url = f"https://www.youtube.com/results?search_query={term.replace(' ', '+')}"
            return "youtube", url, f"Abrindo YouTube para: {term}"
    
    # Google
    if re.search(r'\b(pesquisar|buscar|google|procurar)\b', text_lower):
        term = re.sub(
            r'\b(pesquisar|buscar|google|procurar|sobre|o que é|a|no)\b',
            '', 
            text_lower, 
            flags=re.IGNORECASE
        ).strip()
        
        if term:
            url = f"https://www.google.com/search?q={term.replace(' ', '+')}"
            return "google", url, f"Pesquisando no Google: {term}"
    
    return "chat", None, None

def open_link_js(url):
    """Abre link em nova aba"""
    js_code = f"""
    <script>
        window.open('{url}', '_blank');
    </script>
    """
    components.html(js_code, height=0)

# ═══════════════════════════════════════════════════════════
# 📱 INTERFACE PRINCIPAL
# ═══════════════════════════════════════════════════════════
def main():
    # Inicializa estados
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "app_state" not in st.session_state:
        st.session_state.app_state = "idle"  # idle, listening, processing

    # --- BARRA LATERAL ---
    with st.sidebar:
        st.header("🛠️ Configurações")
        
        input_mode = st.radio(
            "Modo de Entrada:",
            ("🗣️ Voz (Orbe)", "⌨️ Texto (Chat)"),
            index=0
        )
        
        st.divider()
        
        st.markdown("### 🔊 Configurações de Voz")
        voice_accent = st.selectbox(
            "Sotaque:",
            ("Brasil (Padrão)", "Portugal"),
            index=0
        )
        
        silence_threshold = st.slider(
            "Sensibilidade de Silêncio:",
            100, 2000, 500, 50,
            help="Menor = mais sensível"
        )
        
        st.divider()
        
        if st.button("🧹 Limpar Histórico"):
            st.session_state.messages = []
            st.session_state.app_state = "idle"
            st.rerun()

    # --- TÍTULO ---
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 30px;'>"
        "MONKEY<span style='color:#38bdf8;'>AI</span> "
        "<span style='font-size:0.5em; opacity:0.6;'>🐵</span>"
        "</h1>", 
        unsafe_allow_html=True
    )

    user_input = None

    # --- MODO ORBE ---
    if "Voz" in input_mode:
        # Container principal do orbe
        st.markdown('<div class="orb-section">', unsafe_allow_html=True)
        
        # BANNER DE STATUS
        state = st.session_state.app_state
        
        if state == "idle":
            banner_class = ""
            icon = "💤"
            text = "MonkeyAI em Espera"
            subtext = "Clique no orbe para começar"
        elif state == "listening":
            banner_class = "listening"
            icon = "🎤"
            text = "OUVINDO VOCÊ AGORA!"
            subtext = "Fale seu comando... (para automaticamente no silêncio)"
        else:  # processing
            banner_class = "processing"
            icon = "⚙️"
            text = "Processando..."
            subtext = "Aguarde um momento"
        
        st.markdown(f"""
        <div class="status-banner {banner_class}">
            <span class="status-icon">{icon}</span>
            <p class="status-text">{text}</p>
            <p class="status-subtext">{subtext}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # ORBE
        orb_class = "listening" if state == "listening" else ("processing" if state == "processing" else "")
        orb_icon = "🔴" if state == "listening" else ("⚡" if state == "processing" else "🐵")
        
        st.markdown(f"""
        <div class="orb-wrapper">
            <div class="orb-ring"></div>
            <div class="orb-ring"></div>
            <div class="orb-ring"></div>
            <div class="orb-core {orb_class}">
                <div class="orb-icon">{orb_icon}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # BOTÃO DE ATIVAÇÃO
        if state == "idle":
            if st.button("🎙️ ATIVAR MONKEYAI", key="activate", use_container_width=True, type="primary"):
                st.session_state.app_state = "listening"
                st.rerun()
        
        # PROCESSAMENTO DE ÁUDIO
        if state == "listening":
            # Grava áudio
            audio_data = record_audio_with_silence_detection(
                duration=10,
                silence_threshold=silence_threshold
            )
            
            if audio_data:
                st.session_state.app_state = "processing"
                st.rerun()
                
                # Transcreve
                user_input = transcribe_audio(audio_data)
                
                if not user_input:
                    st.toast("⚠️ Não consegui entender. Tente novamente.")
                    st.session_state.app_state = "idle"
                    time.sleep(1)
                    st.rerun()
            else:
                st.session_state.app_state = "idle"
                st.rerun()
    
    # --- MODO TEXTO ---
    else:
        user_input = st.chat_input("Digite seu comando aqui...")

    # --- PROCESSAMENTO DE ENTRADA ---
    if user_input:
        # Adiciona ao histórico
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Verifica comandos
        cmd_type, url, reply_text = check_commands(user_input)
        
        ai_response = ""
        
        if cmd_type != "chat":
            # Comando de redirecionamento
            ai_response = reply_text
            st.session_state.messages.append({"role": "bot", "content": ai_response})
            
            st.toast(f"🚀 {ai_response}")
            open_link_js(url)
            
            # Áudio
            if "Voz" in input_mode:
                audio_fp = text_to_speech_gtts(ai_response, voice_accent)
                if audio_fp:
                    st.audio(audio_fp, format='audio/mp3', autoplay=True)
        else:
            # Conversa com IA
            chain = get_ai_chain()
            hist_text = "\n".join([
                f"{m['role']}: {m['content']}" 
                for m in st.session_state.messages[-6:]
            ])
            ai_response = chain.invoke({"history": hist_text, "input": user_input})
            
            st.session_state.messages.append({"role": "bot", "content": ai_response})
            
            # Áudio
            if "Voz" in input_mode:
                audio_fp = text_to_speech_gtts(ai_response, voice_accent)
                if audio_fp:
                    st.audio(audio_fp, format='audio/mp3', autoplay=True)
        
        # Reseta estado
        st.session_state.app_state = "idle"
        st.rerun()

    # --- HISTÓRICO DE CHAT ---
    if st.session_state.messages:
        st.divider()
        st.markdown("### 💬 Histórico de Conversas")
        
        for msg in st.session_state.messages:
            css_class = "user-message" if msg["role"] == "user" else "bot-message"
            label = "VOCÊ" if msg["role"] == "user" else "MONKEYAI 🐵"
            st.markdown(f"""
                <div class="chat-message {css_class}">
                    <div class="message-label">{label}</div>
                    <div class="message-text">{msg['content']}</div>
                </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
