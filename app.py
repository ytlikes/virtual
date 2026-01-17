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
# ⚙️ CONFIGURAÇÃO & SEGURANÇA
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Jarvis V4",
    page_icon="⚛️",
    layout="centered",
    initial_sidebar_state="expanded" # Barra lateral aberta para ver as opções
)

load_dotenv()
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

if not os.getenv("GROQ_API_KEY"):
    st.error("⚠️ Erro: Chave API Groq não configurada.")
    st.stop()

# ═══════════════════════════════════════════════════════════
# 🎨 CSS AVANÇADO - O ORBE DE ENERGIA
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

    /* --- O ORBE (Estilizando o botão do gravador) --- */
    /* Container para centralizar */
    .orb-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 50vh;
    }

    /* O botão em si */
    .orb-container button {
        width: 160px !important;
        height: 160px !important;
        border-radius: 50% !important;
        border: none !important;
        /* Gradiente estilo reator */
        background: radial-gradient(circle at 30% 30%, rgba(56, 189, 248, 0.8), rgba(3, 105, 161, 1) 60%, rgba(0, 0, 0, 1) 100%) !important;
        /* Múltiplas sombras para o brilho (Glow) */
        box-shadow: 
            0 0 20px rgba(56, 189, 248, 0.6), /* Brilho interno */
            0 0 40px rgba(3, 105, 161, 0.4), /* Brilho médio */
            0 0 80px rgba(2, 132, 199, 0.2), /* Brilho externo */
            inset 0 0 30px rgba(255, 255, 255, 0.3) !important; /* Reflexo */
        
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        color: transparent !important; /* Esconde o ícone padrão */
        animation: pulse-blue 3s infinite ease-in-out;
    }

    /* Efeito ao passar o mouse ou clicar (Fica mais intenso) */
    .orb-container button:hover, .orb-container button:active {
        transform: scale(1.05);
        box-shadow: 
            0 0 30px rgba(56, 189, 248, 0.8),
            0 0 60px rgba(3, 105, 161, 0.6),
            0 0 100px rgba(2, 132, 199, 0.4),
            inset 0 0 40px rgba(255, 255, 255, 0.5) !important;
    }

    @keyframes pulse-blue {
        0%, 100% { opacity: 0.9; transform: scale(1); }
        50% { opacity: 1; transform: scale(1.02); }
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
# 🧠 FUNÇÕES DE BACKEND (IA, Áudio, Comandos)
# ═══════════════════════════════════════════════════════════

@st.cache_resource
def get_ai_chain():
    """Carrega a IA Llama-3 via Groq"""
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.4, max_tokens=150)
    prompt = ChatPromptTemplate.from_template(
        "Você é o Jarvis. Responda de forma breve, inteligente e prestativa.\nHistórico recente:\n{history}\nHuman: {input}\nJarvis:"
    )
    return prompt | llm | StrOutputParser()

def transcribe_audio(audio_bytes):
    """Converte áudio em texto usando Google Speech Recognition"""
    r = sr.Recognizer()
    try:
        with io.BytesIO(audio_bytes) as source_audio:
            with sr.AudioFile(source_audio) as source:
                # Ajusta para ruído ambiente rapidamente
                r.adjust_for_ambient_noise(source, duration=0.5)
                audio_content = r.record(source)
                # Tenta reconhecer
                text = r.recognize_google(audio_content, language='pt-BR')
                return text
    except sr.UnknownValueError:
        return None # Não entendeu
    except sr.RequestError:
        st.error("Erro de conexão com serviço de reconhecimento.")
        return None
    except Exception as e:
        st.error(f"Erro de áudio: {e}")
        return None

def text_to_speech_gtts(text, lang_choice):
    """Gera áudio usando gTTS com seletor de idioma/sotaque"""
    try:
        # Limpa URLs para não ler links
        clean_text = re.sub(r'http\S+', 'link', text)
        
        # Seleção de sotaque (gTTS na nuvem é limitado a isso)
        language = 'pt'
        tld = 'com.br' # Padrão Brasil
        
        if lang_choice == 'Portugal':
            tld = 'pt'
            
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
    if re.search(r'\b(tocar|ouvir|ver|assistir|youtube)\b', text_lower):
        term = re.sub(r'\b(tocar|ouvir|ver|assistir|video|musica|no|youtube|o)\b', '', text_lower, flags=re.IGNORECASE).strip()
        return "youtube", f"https://www.youtube.com/results?search_query={term}", f"Abrindo YouTube para: {term}"
    
    if re.search(r'\b(pesquisar|buscar|google)\b', text_lower):
        term = re.sub(r'\b(pesquisar|buscar|google|sobre|o que é|a)\b', '', text_lower, flags=re.IGNORECASE).strip()
        return "google", f"https://www.google.com/search?q={term}", f"Pesquisando no Google: {term}"
        
    return "chat", None, None

def open_link_js(url):
    """Injeta JS para abrir link"""
    js = f"<script>window.open('{url}', '_blank').focus();</script>"
    components.html(js, height=0)

# ═══════════════════════════════════════════════════════════
# 📱 INTERFACE PRINCIPAL
# ═══════════════════════════════════════════════════════════
def main():
    # Inicializa histórico
    if "messages" not in st.session_state: st.session_state.messages = []

    # --- BARRA LATERAL (CONFIGURAÇÕES) ---
    with st.sidebar:
        st.header("🛠️ Configurações")
        
        # Seletor de Modo de Entrada
        input_mode = st.radio(
            "Modo de Entrada:",
            ("🗣️ Voz (Orbe)", "⌨️ Texto (Chat)"),
            index=0
        )
        
        st.divider()
        
        # Seletor de Voz (Sotaque - Limitação do gTTS Grátis)
        st.markdown("### 🔊 Seleção de Voz")
        st.info("Nota: Na nuvem gratuita, a variação se limita a sotaques regionais.")
        voice_accent = st.selectbox(
            "Sotaque da Voz:",
            ("Brasil (Padrão)", "Portugal"),
            index=0
        )
        
        st.divider()
        if st.button("🧹 Limpar Histórico"):
            st.session_state.messages = []
            st.rerun()

    # --- TÍTULO ---
    st.markdown("<h1 style='text-align: center; margin-bottom: 30px;'>JARVIS <span style='color:#38bdf8; font-size:0.6em;'>V4</span></h1>", unsafe_allow_html=True)

    user_input = None

    # --- LÓGICA DE EXIBIÇÃO (ORBE OU TEXTO) ---
    if "Voz" in input_mode:
        # MODO ORBE LIGADO
        st.markdown('<div class="orb-container">', unsafe_allow_html=True)
        # O gravador que vira o Orbe via CSS
        audio_data = mic_recorder(
            start_prompt="⚫", # Ícones transparentes/irrelevantes pois o CSS cobre
            stop_prompt="🔴", 
            key='orb_recorder',
            format="wav",
            use_container_width=False
        )
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; opacity: 0.6; margin-top: -20px;'>Toque no núcleo para ativar</p>", unsafe_allow_html=True)

        # Processa áudio se houver gravação
        if audio_data:
            with st.spinner("Processando áudio..."):
                user_input = transcribe_audio(audio_data['bytes'])
                if not user_input:
                    st.toast("⚠️ Não entendi o áudio. Tente novamente.")
    else:
        # MODO TEXTO LIGADO (Chat padrão no fundo)
        user_input = st.chat_input("Digite seu comando aqui...")


    # --- NÚCLEO DE PROCESSAMENTO (Se houver entrada) ---
    if user_input:
        # 1. Adiciona ao histórico visual
        st.session_state.messages.append({"role": "user", "content": user_input})

        # 2. Verifica Comandos
        cmd_type, url, reply_text = check_commands(user_input)
        
        ai_response = ""
        
        if cmd_type != "chat":
            # É comando (YouTube/Google)
            ai_response = reply_text
            st.toast(f"🚀 {ai_response}")
            open_link_js(url)
        else:
            # É conversa normal (Chama a IA)
            with st.spinner("Jarvis pensando..."):
                chain = get_ai_chain()
                # Formata histórico recente para a IA
                hist_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-4:]])
                ai_response = chain.invoke({"history": hist_text, "input": user_input})
        
        # 3. Adiciona resposta ao histórico
        st.session_state.messages.append({"role": "bot", "content": ai_response})
        
        # 4. Gera e Toca Áudio (Se estiver no modo voz)
        if "Voz" in input_mode:
            accent_code = 'Portugal' if voice_accent == 'Portugal' else 'Brasil'
            audio_fp = text_to_speech_gtts(ai_response, accent_code)
            if audio_fp:
                 # Autoplay invisível para fluidez
                st.audio(audio_fp, format='audio/mp3', autoplay=True)

        # Rerun para atualizar o chat se for entrada de texto
        if "Texto" in input_mode:
            st.rerun()


    # --- EXIBIÇÃO DO HISTÓRICO DE CHAT ---
    st.divider()
    for msg in st.session_state.messages:
        css_class = "user-message" if msg["role"] == "user" else "bot-message"
        label = "VOCÊ" if msg["role"] == "user" else "JARVIS"
        st.markdown(f"""
            <div class="chat-message {css_class}">
                <div class="message-label">{label}</div>
                <div class="message-text">{msg['content']}</div>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
