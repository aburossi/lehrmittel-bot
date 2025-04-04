import streamlit as st
import google.generativeai as genai
import os
from pathlib import Path
import re

# --- Configuration ---

st.set_page_config(
    page_title="Subchapter Chatbot",
    layout="wide"
)

# Get API Key from Streamlit secrets using the .get() method
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY not found or not set in Streamlit secrets. Please add it.")
    st.stop()

try:
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"Error configuring Google Generative AI: {e}")
    st.stop()

# LearnLM Model Configuration
generation_config = {
    "temperature": 0.4,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Define the directory containing textbooks
TEXTBOOK_DIR = Path("textbooks")
PLACEHOLDER_SELECT = "-- Select a Subchapter --" # Define placeholder

# --- Helper Functions ---

def get_available_subchapters(textbook_dir: Path) -> dict[str, str]:
    """
    Scans the textbook directory for files matching the pattern
    'MainChapter_Topic_SubChapter.txt', creates a display name,
    and returns a dictionary mapping {display_name: filename}.
    """
    subchapter_map = {}
    if not textbook_dir.is_dir():
        return subchapter_map

    for f in textbook_dir.glob("*.txt"):
        filename = f.name
        stem = f.stem
        parts = stem.split('_')
        if len(parts) == 3:
            main_chapter_str, topic_str, subchapter_str = parts
            display_name = subchapter_str
            subchapter_map[display_name] = filename
        else:
             print(f"Info: Skipping file with unexpected name format: {filename}")

    sorted_subchapter_map = dict(sorted(subchapter_map.items()))
    return sorted_subchapter_map

def load_subchapter_content(filename: str, textbook_dir: Path) -> str | None:
    """Loads the content of a specific subchapter file using its filename."""
    file_path = textbook_dir / filename
    if file_path.is_file():
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            st.error(f"Error reading subchapter file '{filename}': {e}")
            return None
    else:
        st.error(f"Subchapter file not found: {file_path}")
        return None

def initialize_learnlm_model(system_prompt: str) -> genai.GenerativeModel | None:
    """Initializes the GenerativeModel with a system instruction."""
    try:
        model = genai.GenerativeModel(
            model_name="learnlm-1.5-pro-experimental",
            generation_config=generation_config,
            system_instruction=system_prompt,
        )
        return model
    except Exception as e:
        st.error(f"Error initializing LearnLM model: {e}")
        return None

# --- Reset Function ---
def reset_chat_state():
    """Clears chat history and related session state variables."""
    st.session_state.messages = []
    st.session_state.learnlm_model = None
    st.session_state.chat_session = None
    st.session_state.subchapter_content = None
    # Keep selected_subchapter_display_name as is until a new one is chosen or placeholder selected
    print("Chat state reset.") # Optional debug message


# --- Streamlit App UI and Logic ---

st.title("📚 Subchapter Exam Prep Chatbot")
st.caption("Powered by Google LearnLM 1.5 Pro Experimental")

# --- State Initialization ---
# Initialize only if keys don't exist
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_subchapter_display_name" not in st.session_state:
    # Initialize with placeholder to prevent initial load
    st.session_state.selected_subchapter_display_name = PLACEHOLDER_SELECT
if "subchapter_content" not in st.session_state:
    st.session_state.subchapter_content = None
if "learnlm_model" not in st.session_state:
    st.session_state.learnlm_model = None
if "chat_session" not in st.session_state:
    st.session_state.chat_session = None
if "subchapter_map" not in st.session_state:
    st.session_state.subchapter_map = get_available_subchapters(TEXTBOOK_DIR)


# --- Subchapter Selection ---
if not st.session_state.subchapter_map:
    st.warning(
        f"No valid subchapter files (e.g., '8_Topic_8.3 Subtopic.txt') found in the "
        f"'{TEXTBOOK_DIR}' directory. Please check the folder and filenames."
    )
    st.stop()

# Prepare options for the selectbox, including the placeholder
available_display_names = [PLACEHOLDER_SELECT] + list(st.session_state.subchapter_map.keys())

# Determine the index for the selectbox based on the current state
try:
    # Find index of the currently loaded/selected subchapter
    current_index = available_display_names.index(st.session_state.selected_subchapter_display_name)
except ValueError:
    # Default to placeholder if current value isn't in the list (shouldn't happen often)
    current_index = 0

# Store the previous selection to detect changes accurately
previous_selection = st.session_state.selected_subchapter_display_name

selected_display_name = st.selectbox(
    "Select the subchapter you want to study:",
    options=available_display_names,
    key="subchapter_selector",
    index=current_index,
)

# --- Load Subchapter and Initialize Model/Chat ---
# This logic now runs only when the selection changes *and* it's not the placeholder
if selected_display_name != previous_selection:
    st.session_state.selected_subchapter_display_name = selected_display_name # Update state immediately

    if selected_display_name == PLACEHOLDER_SELECT:
        # User selected the placeholder - clear chat state but don't load
        reset_chat_state()
        st.info("Please select a subchapter from the list to begin.")
        # Force rerun to clear chat display if needed (often handled by Streamlit automatically)
        # st.rerun() # Uncomment if chat doesn't clear reliably

    else:
        # User selected a real subchapter - Proceed with loading logic
        st.info(f"Loading subchapter: {selected_display_name}...")
        reset_chat_state() # Clear previous chat before loading new one

        filename_to_load = st.session_state.subchapter_map.get(selected_display_name)

        if filename_to_load:
            content = load_subchapter_content(filename_to_load, TEXTBOOK_DIR)
            if content:
                st.session_state.subchapter_content = content # Store content

                # Define the system prompt
                system_prompt = f"""Du bist ein KI-gestützter Tutor auf Basis von LearnLM und hilfst einem Lernenden dabei, den Inhalt des folgenden Kapitels aus dem Lehrmittel Allgemeinbildung zu verstehen.

                Dein Wissen ist AUSSCHLIESSLICH auf den folgenden Text zum Kapitel '{selected_display_name}' beschränkt. Verwende KEINE externen Informationen und zitiere NIEMALS Textpassagen wortwörtlich – formuliere immer mit eigenen Worten um.

                --- START DES TEXTES ZUM KAPITEL '{selected_display_name}' ---
                {st.session_state.subchapter_content}
                --- ENDE DES TEXTES ZUM KAPITEL '{selected_display_name}' ---

                Wichtige Informationen zum Text:
                - Das Lehrmittel heißt **"Lehrmittel Allgemeinbildung"**
                - Seitenzahlen sind im Format **[seite: XXX]** im Text enthalten
                - Verwende Seitenzahlen strategisch:
                - Gib die relevante Seite an, wenn du ein Thema erklärst oder ein Konzept vertiefst
                - Nutze Seitenzahlen, um den Lernenden zu motivieren zuerst etwas zu lesen oder im Nachhinein nachzuschlagen
                - Nutze Seitenverweise als Lernstrategie („Lies zuerst S.220, dann beantworte die Frage“ oder „Versuche die Frage zu beantworten, danach lies auf S.223 nach“)

                Du arbeitest mit folgenden Prinzipien der Lernwissenschaft:
                - **Aktives Lernen**: Stelle Fragen, rege zum Nachdenken und Mitmachen an
                - **Kognitive Entlastung**: Gib nur eine Information oder Aufgabe pro Antwort
                - **Neugier fördern**: Verwende Analogien, stelle interessante Fragen, verbinde Inhalte
                - **Anpassung**: Passe dein Vorgehen an das Niveau und Ziel des Lernenden an
                - **Metakognition**: Fördere Selbstreflexion und Lernbewusstsein

                Sprache: **ANTWORTE AUSSCHLIESSLICH AUF DEUTSCH**

                Beginne das Gespräch mit einer freundlichen Begrüßung und biete folgende Lernmodi an:

                1. 📚 **Quiz mich** – Teste mein Wissen
                2. 💡 **Erkläre ein Konzept**
                3. 🔄 **Verwende eine Analogie**
                4. 🔍 **Gehe tiefer auf ein Thema ein**
                5. 🧠 **Reflektiere oder fasse zusammen**
                6. 🧩 **Erstelle eine Konzeptkarte**

                Warte, bis sich der Lernende für einen Modus entscheidet.

                Spezifisches Verhalten je nach Modus:

                - **📚 Quiz mich**: Stelle 1 Frage pro Durchlauf, beginnend einfach, dann steigend. Bitte um Begründung der Antwort. Wenn korrekt: loben. Wenn falsch: behutsam zur richtigen Lösung führen. Nach 5 Fragen: Zusammenfassung oder Fortsetzung anbieten. Verwende relevante Seitenangaben bei Bedarf (z. B. „Diese Info findest du auf [seite: 221]“).

                - **💡 Erkläre ein Konzept**: Frage zuerst, welches Konzept erklärt werden soll. Gib eine schrittweise Erklärung. Biete relevante Seitenangaben zum Nachlesen an.

                - **🔄 Verwende eine Analogie**: Wähle eine geeignete Stelle im Text aus und erkläre sie mithilfe eines kreativen, aber passenden Vergleichs. Nutze Seitenangaben zur Orientierung.

                - **🔍 Gehe tiefer auf ein Thema ein**: Wenn der Lernende tiefer verstehen möchte, stelle offene, leitende Fragen. Nutze Seitenangaben zur Vertiefung.

                - **🧠 Reflektiere oder fasse zusammen**: Fasse in eigenen Worten zusammen, was besprochen wurde. Stelle Reflexionsfragen wie: „Was fiel dir leicht? Wo möchtest du noch mehr üben?“ Gib ggf. Hinweise auf Seiten zum Wiederholen.

                - **🧩 Konzeptkarte erstellen**: Bitte den Lernenden, 3–5 zentrale Ideen aus dem Kapitel zu nennen. Hilf, Zusammenhänge zu erkennen. Nutze Seitenzahlen zur Verankerung im Text.

                Stil: Sei stets freundlich, unterstützend und geduldig. Stelle pro Antwort nur eine Frage oder Information. Fördere ein Gefühl von Fortschritt und Selbstwirksamkeit.

                Bereit, mit dem Kapitel '{selected_display_name}' aus dem Lehrmittel Allgemeinbildung zu starten? Bitte den Lernenden, einen der 6 Lernmodi auszuwählen.
                """



                # Initialize the model
                st.session_state.learnlm_model = initialize_learnlm_model(system_prompt)

                if st.session_state.learnlm_model:
                    # Start chat session
                    try:
                        st.session_state.chat_session = st.session_state.learnlm_model.start_chat(history=[])
                        st.success(f"Subchapter '{selected_display_name}' loaded. Ask me anything about it!")
                        # Optional initial greeting
                        try:
                             initial_user_message = f"Please introduce yourself..." # As before
                             initial_response = st.session_state.chat_session.send_message(initial_user_message)
                             st.session_state.messages.append({"role": "assistant", "content": initial_response.text})
                        except Exception as e:
                            st.warning(f"Could not get initial greeting from LearnLM: {e}")
                            st.session_state.messages.append({"role": "assistant", "content": f"Hello! I'm ready to help..."}) # As before

                        # Force rerun to update display immediately after loading and initial message
                        st.rerun()

                    except Exception as e:
                         st.error(f"Failed to start chat session: {e}")
                         reset_chat_state() # Reset on failure
                         st.session_state.selected_subchapter_display_name = PLACEHOLDER_SELECT # Revert selection state

                else:
                    # Handle model initialization failure
                    st.error("Failed to initialize the LearnLM model.")
                    reset_chat_state()
                    st.session_state.selected_subchapter_display_name = PLACEHOLDER_SELECT

            else:
                 # Handle content loading failure
                 st.error(f"Failed to load content for {selected_display_name}.")
                 reset_chat_state()
                 st.session_state.selected_subchapter_display_name = PLACEHOLDER_SELECT

        else:
            st.error(f"Internal error: Could not find filename for '{selected_display_name}'.")
            reset_chat_state()
            st.session_state.selected_subchapter_display_name = PLACEHOLDER_SELECT


# --- Display Chat History ---
st.markdown("---")
# Show subheader based on whether a real subchapter is selected
current_topic = st.session_state.selected_subchapter_display_name \
                if st.session_state.selected_subchapter_display_name != PLACEHOLDER_SELECT \
                else "No Subchapter Selected"
st.subheader(f"Chatting about: {current_topic}")

# Display messages only if a real subchapter is loaded and chat session exists
if st.session_state.selected_subchapter_display_name != PLACEHOLDER_SELECT and st.session_state.chat_session:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
elif st.session_state.selected_subchapter_display_name == PLACEHOLDER_SELECT and not st.session_state.messages:
    # Show placeholder text if nothing is selected
     st.info("Select a subchapter from the dropdown menu above to start chatting.")


# --- Handle User Input ---
# Disable input if no real subchapter is selected or chat session isn't ready
prompt_disabled = (st.session_state.selected_subchapter_display_name == PLACEHOLDER_SELECT or
                   not st.session_state.chat_session)

user_prompt = st.chat_input(
    f"Ask a question about {st.session_state.selected_subchapter_display_name}..." if not prompt_disabled else "Select a subchapter to enable chat",
    disabled=prompt_disabled,
    key="user_chat_input"
)

if user_prompt:
    # Add user message to state and display it
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Send user message to LearnLM and get response
    try:
        with st.spinner("Thinking..."):
            response = st.session_state.chat_session.send_message(user_prompt)
        assistant_response = response.text
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

        # Rerun needed to display the latest assistant message immediately after sending
        st.rerun()

    except Exception as e:
        st.error(f"An error occurred while communicating with LearnLM: {e}")
        error_message = f"Sorry, I encountered an error: {e}"
        # Add error message to chat history too
        st.session_state.messages.append({"role": "assistant", "content": error_message})
        with st.chat_message("assistant"):
             st.markdown(error_message)
        # Rerun to show the error message in the chat
        st.rerun()

# Optional: Add a button to explicitly clear chat if needed
# if st.button("Clear Current Chat"):
#     reset_chat_state()
#     st.session_state.selected_subchapter_display_name = PLACEHOLDER_SELECT # Reset selection too
#     st.rerun()