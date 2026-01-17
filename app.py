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
# 🎨 CSS AVANÇADO - O ORBE DE ENERGIA APRIMORADO
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

    /* --- O ORBE INTERATIVO --- */
    .orb-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 50vh;
        position: relative;
    }

    .orb-wrapper {
        position: relative;
        width: 180px;
        height: 180px;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    /* Anéis externos animados */
    .orb-ring {
        position: absolute;
        border-radius: 50%;
        border: 2px solid rgba(56, 189, 248, 0.3);
        animation: ring-pulse 3s infinite ease-in-out;
    }

    .orb-ring:nth-child(1) {
        width: 200px;
        height: 200px;
        animation-delay: 0s;
    }

    .orb-ring:nth-child(2) {
        width: 240px;
        height: 240px;
        animation-delay: 0.5s;
    }

    .orb-ring:nth-child(3) {
        width: 280px;
        height: 280px;
        animation-delay: 1s;
    }

    @keyframes ring-pulse {
        0%, 100% { 
            opacity: 0.2; 
            transform: scale(0.95);
        }
        50% { 
            opacity: 0.6; 
            transform: scale(1.05);
        }
    }

    /* O núcleo do orbe */
    .orb-core {
        width: 160px;
        height: 160px;
        border-radius: 50%;
        background: radial-gradient(circle at 30% 30%, 
            rgba(56, 189, 248, 0.9), 
            rgba(3, 105, 161, 1) 50%, 
            rgba(0, 0, 0, 1) 100%);
        box-shadow: 
            0 0 30px rgba(56, 189, 248, 0.8),
            0 0 60px rgba(3, 105, 161, 0.5),
            0 0 100px rgba(2, 132, 199, 0.3),
            inset 0 0 40px rgba(255, 255, 255, 0.4);
        cursor: pointer;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        z-index: 10;
        display: flex;
        justify-content: center;
        align-items: center;
        animation: idle-pulse 4s infinite ease-in-out;
    }

    .orb-core:hover {
        transform: scale(1.08);
        box-shadow: 
            0 0 40px rgba(56, 189, 248, 1),
            0 0 80px rgba(3, 105, 161, 0.7),
            0 0 120px rgba(2, 132, 199, 0.5),
            inset 0 0 50px rgba(255, 255, 255, 0.6);
    }

    /* Estado ao clicar - Brilho intenso */
    .orb-core:active {
        transform: scale(1.12);
        box-shadow: 
            0 0 60px rgba(56, 189, 248, 1),
            0 0 100px rgba(3, 105, 161, 0.9),
            0 0 150px rgba(2, 132, 199, 0.7),
            inset 0 0 60px rgba(255, 255, 255, 0.8);
        filter: brightness(1.3);
    }

    /* Estado de gravação ativa */
    .orb-core.recording {
        animation: recording-pulse 1s infinite ease-in-out;
        background: radial-gradient(circle at 30% 30%, 
            rgba(239, 68, 68, 0.9), 
            rgba(185, 28, 28, 1) 50%, 
            rgba(0, 0, 0, 1) 100%);
        box-shadow: 
            0 0 40px rgba(239, 68, 68, 0.9),
            0 0 80px rgba(185, 28, 28, 0.6),
            0 0 120px rgba(153, 27, 27, 0.4);
    }

    /* Estado desligado/idle com menos brilho */
    .orb-core.idle {
        box-shadow: 
            0 0 15px rgba(56, 189, 248, 0.4),
            0 0 30px rgba(3, 105, 161, 0.2),
            0 0 50px rgba(2, 132, 199, 0.1),
            inset 0 0 20px rgba(255, 255, 255, 0.2);
        opacity: 0.8;
    }

    /* Estado de processamento */
    .orb-core.processing {
        animation: processing-spin 2s infinite linear;
        background: radial-gradient(circle at 30% 30%, 
            rgba(168, 85, 247, 0.9), 
            rgba(126, 34, 206, 1) 50%, 
            rgba(0, 0, 0, 1) 100%);
        box-shadow: 
            0 0 40px rgba(168, 85, 247, 0.9),
            0 0 80px rgba(126, 34, 206, 0.6),
            0 0 120px rgba(107, 33, 168, 0.4);
    }

    @keyframes idle-pulse {
        0%, 100% { 
            opacity: 0.95; 
            transform: scale(1);
        }
        50% { 
            opacity: 1; 
            transform: scale(1.03);
        }
    }

    @keyframes recording-pulse {
        0%, 100% { 
            transform: scale(1); 
            opacity: 1;
        }
        50% { 
            transform: scale(1.06); 
            opacity: 0.9;
        }
    }

    @keyframes processing-spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    /* Ícone no centro do orbe */
    .orb-icon {
        font-size: 48px;
        color: rgba(255, 255, 255, 0.9);
        text-shadow: 0 0 20px rgba(255, 255, 255, 0.5);
    }

    /* Status text */
    .orb-status {
        margin-top: 30px;
        text-align: center;
        font-size: 18px;
        color: #38bdf8;
        opacity: 0.8;
        min-height: 30px;
    }

    /* --- ESTILO DO CHAT --- */
    .chat-message {
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: rgba(56, 189, 248, 0.1);
        border-left: 4px solid #38bdf8;
        align-items: flex-end;
    }
    .bot-message {
        background-color: rgba(31, 41, 55, 0.6);
        border: 1px solid rgba(56, 189, 248, 0.2);
        align-items: flex-start;
    }
    .message-text {
        font-family: 'Inter', sans-serif;
        font-size: 16px;
    }
    .message-label {
        font-size: 12px;
        opacity: 0.7;
        margin-bottom: 4px;
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

def record_audio_with_silence_detection(duration=10, silence_threshold=500, silence_duration=2):
    """
    Grava áudio do microfone e para automaticamente quando detectar silêncio.
    
    Args:
        duration: Tempo máximo de gravação em segundos
        silence_threshold: Limiar de energia para considerar silêncio (ajuste conforme necessário)
        silence_duration: Tempo de silêncio em segundos para parar a gravação
    """
    r = sr.Recognizer()
    r.energy_threshold = silence_threshold
    r.dynamic_energy_threshold = True
    
    try:
        with sr.Microphone() as source:
            st.toast("🎤 Ajustando para ruído ambiente...")
            r.adjust_for_ambient_noise(source, duration=1)
            
            st.toast("🔴 Gravando... Fale agora!")
            
            # Grava com detecção de silêncio
            audio_data = r.listen(
                source, 
                timeout=duration,
                phrase_time_limit=duration
            )
            
            return audio_data
            
    except sr.WaitTimeoutError:
        st.warning("⏱️ Tempo limite de gravação atingido.")
        return None
    except Exception as e:
        st.error(f"❌ Erro ao gravar áudio: {e}")
        return None

def transcribe_audio(audio_data):
    """Converte áudio em texto usando Google Speech Recognition"""
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
    
    # Detecta comando YouTube
    if re.search(r'\b(tocar|ouvir|ver|assistir|youtube|vídeo|música)\b', text_lower):
        # Remove palavras de comando para extrair o termo de busca
        term = re.sub(
            r'\b(tocar|ouvir|ver|assistir|video|vídeo|musica|música|no|youtube|o|a|de|da|do)\b',
            '', 
            text_lower, 
            flags=re.IGNORECASE
        ).strip()
        
        if term:
            url = f"https://www.youtube.com/results?search_query={term.replace(' ', '+')}"
            return "youtube", url, f"🎵 Abrindo YouTube para: {term}"
    
    # Detecta comando Google
    if re.search(r'\b(pesquisar|buscar|google|procurar)\b', text_lower):
        term = re.sub(
            r'\b(pesquisar|buscar|google|procurar|sobre|o que é|a|no)\b',
            '', 
            text_lower, 
            flags=re.IGNORECASE
        ).strip()
        
        if term:
            url = f"https://www.google.com/search?q={term.replace(' ', '+')}"
            return "google", url, f"🔍 Pesquisando no Google: {term}"
    
    return "chat", None, None

def open_link_js(url):
    """Abre link em nova aba usando JavaScript"""
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
    if "orb_state" not in st.session_state:
        st.session_state.orb_state = "idle"  # idle, recording, processing
    if "trigger_recording" not in st.session_state:
        st.session_state.trigger_recording = False

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
            help="Menor = mais sensível ao silêncio"
        )
        
        st.divider()
        
        if st.button("🧹 Limpar Histórico"):
            st.session_state.messages = []
            st.rerun()

    # --- TÍTULO ---
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 20px;'>"
        "MONKEY<span style='color:#38bdf8;'>AI</span> "
        "<span style='font-size:0.5em; opacity:0.6;'>🐵</span>"
        "</h1>", 
        unsafe_allow_html=True
    )

    user_input = None

    # --- MODO ORBE ---
    if "Voz" in input_mode:
        # Container do orbe
        orb_col = st.container()
        
        with orb_col:
            # Estado do orbe
            orb_class = st.session_state.orb_state
            
            if orb_class == "idle":
                status_text = "Toque no núcleo para ativar"
                icon = "🎙️"
            elif orb_class == "recording":
                status_text = "🔴 Gravando... Fale agora!"
                icon = "⚫"
            else:  # processing
                status_text = "⚙️ Processando..."
                icon = "⚡"
            
            # HTML do orbe
            st.markdown(f"""
            <div class="orb-container">
                <div class="orb-wrapper">
                    <div class="orb-ring"></div>
                    <div class="orb-ring"></div>
                    <div class="orb-ring"></div>
                    <div class="orb-core {orb_class}">
                        <div class="orb-icon">{icon}</div>
                    </div>
                </div>
                <div class="orb-status">{status_text}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Botão invisível para trigger com efeito de clique
            clicked = st.button("🎤 Ativar MonkeyAI", key="orb_trigger", use_container_width=True)
            
            # Adiciona efeito visual de clique com JavaScript
            if clicked:
                st.session_state.trigger_recording = True
                st.session_state.orb_state = "recording"
                
                # Injeta CSS temporário para efeito de clique
                components.html("""
                <script>
                    // Encontra o orbe e adiciona efeito de flash
                    setTimeout(() => {
                        const style = document.createElement('style');
                        style.innerHTML = `
                            .orb-core {
                                animation: click-flash 0.3s ease-out !important;
                            }
                            @keyframes click-flash {
                                0% { transform: scale(1); filter: brightness(1); }
                                50% { transform: scale(1.15); filter: brightness(1.5); }
                                100% { transform: scale(1); filter: brightness(1); }
                            }
                        `;
                        document.head.appendChild(style);
                    }, 100);
                </script>
                """, height=0)
                
                st.rerun()
        
        # Processamento de gravação
        if st.session_state.trigger_recording:
            st.session_state.trigger_recording = False
            
            # Grava com detecção de silêncio
            audio_data = record_audio_with_silence_detection(
                duration=10,
                silence_threshold=silence_threshold,
                silence_duration=2
            )
            
            if audio_data:
                st.session_state.orb_state = "processing"
                st.rerun()
                
                # Transcreve
                user_input = transcribe_audio(audio_data)
                
                if not user_input:
                    st.toast("⚠️ Não consegui entender. Tente novamente.")
                    st.session_state.orb_state = "idle"
                    st.rerun()
            else:
                st.session_state.orb_state = "idle"
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
            
            # Abre link
            st.toast(f"🚀 {ai_response}")
            open_link_js(url)
            
            # Áudio de confirmação
            if "Voz" in input_mode:
                audio_fp = text_to_speech_gtts(ai_response, voice_accent)
                if audio_fp:
                    st.audio(audio_fp, format='audio/mp3', autoplay=True)
        else:
            # Conversa normal com IA
            with st.spinner("🧠 Jarvis pensando..."):
                chain = get_ai_chain()
                hist_text = "\n".join([
                    f"{m['role']}: {m['content']}" 
                    for m in st.session_state.messages[-6:]
                ])
                ai_response = chain.invoke({"history": hist_text, "input": user_input})
            
            st.session_state.messages.append({"role": "bot", "content": ai_response})
            
            # Gera áudio
            if "Voz" in input_mode:
                audio_fp = text_to_speech_gtts(ai_response, voice_accent)
                if audio_fp:
                    st.audio(audio_fp, format='audio/mp3', autoplay=True)
        
        # Reseta estado do orbe
        st.session_state.orb_state = "idle"
        st.rerun()

    # --- HISTÓRICO DE CHAT ---
    if st.session_state.messages:
        st.divider()
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
