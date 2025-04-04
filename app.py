import streamlit as st
import google.generativeai as genai
import os
import boto3
from botocore.exceptions import ClientError
from pathlib import Path

# --- Configuration ---
st.set_page_config(
    page_title="Subchapter Chatbot (AWS S3)",
    layout="wide"
)

# --- Secret Management & Client Initialization ---
try:
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
    aws_access_key_id = st.secrets["AWS_ACCESS_KEY_ID"]
    aws_secret_access_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
    s3_bucket_name = st.secrets["S3_BUCKET_NAME"]
    aws_region = st.secrets.get("AWS_REGION", "us-east-1")  # Default to us-east-1 if not specified

    # Validate presence
    if not gemini_api_key or not aws_access_key_id or not aws_secret_access_key or not s3_bucket_name:
        st.error("One or more required secrets are missing.")
        st.stop()

except KeyError as e:
    st.error(f"Missing required Streamlit secret: {e}. Please configure secrets.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred loading secrets: {e}")
    st.stop()

# Configure Gemini
try:
    genai.configure(api_key=gemini_api_key)
except Exception as e:
    st.error(f"Error configuring Google Generative AI: {e}")
    st.stop()

# --- AWS S3 Client Initialization (Cached) ---
@st.cache_resource(show_spinner="Connecting to AWS S3...")
def get_s3_client():
    """Initializes and returns an S3 client using credentials from secrets."""
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region
        )
        
        # Test connection by trying to get the bucket
        s3_client.head_bucket(Bucket=s3_bucket_name)
        print("Successfully connected to S3.")
        return s3_client
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        if error_code == '404':
            st.error(f"S3 Bucket '{s3_bucket_name}' not found. Check the bucket name.")
        elif error_code == '403':
            st.error(f"Access denied to S3 Bucket '{s3_bucket_name}'. Check IAM permissions.")
        else:
            st.error(f"Error connecting to S3: {e}")
        return None
    except Exception as e:
        st.error(f"Error initializing S3 client: {e}")
        return None

s3_client = get_s3_client()
if not s3_client:
    st.stop()  # Stop if client initialization failed

# LearnLM Model Configuration
generation_config = {
    "temperature": 0.4,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

PLACEHOLDER_SELECT = "-- Select a Subchapter --"

# --- S3 Helper Functions (Cached) ---

@st.cache_data(show_spinner="Listing available subchapters...")
def get_available_subchapters_from_s3(bucket_name: str, _client) -> dict[str, str]:
    """
    Lists objects in the S3 bucket, parses names matching the pattern,
    and returns a dictionary mapping {display_name: object_key}.
    """
    subchapter_map = {}
    try:
        # List all objects in the bucket
        response = _client.list_objects_v2(Bucket=bucket_name)
        
        if 'Contents' in response:
            for obj in response['Contents']:
                object_key = obj['Key']
                # Handle potential folder structure in object key
                filename = Path(object_key).name
                stem = Path(filename).stem

                parts = stem.split('_')
                if len(parts) == 3 and filename.endswith(".txt"):
                    main_chapter_str, topic_str, subchapter_str = parts
                    display_name = subchapter_str
                    # Use the full object key as the value
                    subchapter_map[display_name] = object_key
                else:
                    print(f"Info: Skipping object with unexpected name format: {object_key}")
                    
        # Handle pagination if there are more than 1000 objects
        while response.get('IsTruncated', False):
            response = _client.list_objects_v2(
                Bucket=bucket_name,
                ContinuationToken=response['NextContinuationToken']
            )
            if 'Contents' in response:
                for obj in response['Contents']:
                    object_key = obj['Key']
                    filename = Path(object_key).name
                    stem = Path(filename).stem

                    parts = stem.split('_')
                    if len(parts) == 3 and filename.endswith(".txt"):
                        main_chapter_str, topic_str, subchapter_str = parts
                        display_name = subchapter_str
                        subchapter_map[display_name] = object_key
                    else:
                        print(f"Info: Skipping object with unexpected name format: {object_key}")

    except ClientError as e:
        st.error(f"Error listing files in S3 Bucket '{bucket_name}': {e}")
        return {}
    except Exception as e:
        st.error(f"An unexpected error occurred listing S3 files: {e}")
        return {}

    sorted_subchapter_map = dict(sorted(subchapter_map.items()))
    return sorted_subchapter_map

@st.cache_data(show_spinner="Loading subchapter content...")
def load_subchapter_content_from_s3(bucket_name: str, object_key: str, _client) -> str | None:
    """Loads the content of a specific object from S3."""
    try:
        response = _client.get_object(Bucket=bucket_name, Key=object_key)
        content = response['Body'].read().decode('utf-8')
        return content
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        if error_code == 'NoSuchKey':
            st.error(f"Subchapter file '{object_key}' not found in S3 bucket '{bucket_name}'.")
        elif error_code == 'AccessDenied':
            st.error(f"Access denied to file '{object_key}'. Check IAM permissions.")
        else:
            st.error(f"Error accessing S3 file '{object_key}': {e}")
        return None
    except Exception as e:
        st.error(f"An error occurred reading file '{object_key}' from S3: {e}")
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
    print("Chat state reset.")

# --- Streamlit App UI and Logic ---

st.title("📚 Subchapter Exam Prep Chatbot (AWS S3)")
st.caption("Powered by Google LearnLM 1.5 Pro Experimental | Content from AWS S3")

# --- State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_subchapter_display_name" not in st.session_state:
    st.session_state.selected_subchapter_display_name = PLACEHOLDER_SELECT
if "subchapter_content" not in st.session_state:
    st.session_state.subchapter_content = None
if "learnlm_model" not in st.session_state:
    st.session_state.learnlm_model = None
if "chat_session" not in st.session_state:
    st.session_state.chat_session = None
if "subchapter_map" not in st.session_state:
    # Load map using the S3 function and cached client
    st.session_state.subchapter_map = get_available_subchapters_from_s3(s3_bucket_name, s3_client)

# --- Subchapter Selection ---
if not st.session_state.subchapter_map:
    st.warning(
        f"No valid subchapter files found in S3 bucket '{s3_bucket_name}' or failed to list them. "
        f"Ensure files exist and follow the 'Main_Topic_Subtopic.txt' format, and check permissions."
    )

# Prepare options, including the placeholder
available_display_names = [PLACEHOLDER_SELECT] + list(st.session_state.subchapter_map.keys())

try:
    current_index = available_display_names.index(st.session_state.selected_subchapter_display_name)
except ValueError:
    current_index = 0

previous_selection = st.session_state.selected_subchapter_display_name

selected_display_name = st.selectbox(
    "Select the subchapter you want to study:",
    options=available_display_names,
    key="subchapter_selector",
    index=current_index,
)

# --- Load Subchapter and Initialize Model/Chat ---
if selected_display_name != previous_selection:
    st.session_state.selected_subchapter_display_name = selected_display_name

    if selected_display_name == PLACEHOLDER_SELECT:
        reset_chat_state()
        st.info("Please select a subchapter from the list to begin.")
        st.rerun()  # Ensure UI updates fully after reset

    else:
        st.info(f"Loading subchapter: {selected_display_name} from S3...")
        reset_chat_state()

        # Get the corresponding object key
        object_key_to_load = st.session_state.subchapter_map.get(selected_display_name)

        if object_key_to_load:
            # Load content using the S3 function and cached client
            content = load_subchapter_content_from_s3(s3_bucket_name, object_key_to_load, s3_client)

            if content:  # Check if content loading was successful
                st.session_state.subchapter_content = content

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
                        st.success(f"Subchapter '{selected_display_name}' loaded from S3. Ask me anything about it!")
                        # Optional initial greeting
                        try:
                            initial_user_message = f"Please introduce yourself..."
                            initial_response = st.session_state.chat_session.send_message(initial_user_message)
                            st.session_state.messages.append({"role": "assistant", "content": initial_response.text})
                        except Exception as e:
                            st.warning(f"Could not get initial greeting from LearnLM: {e}")
                            st.session_state.messages.append({"role": "assistant", "content": f"Hello! I'm ready to help you understand the concepts in '{selected_display_name}'. Feel free to ask me any questions about this subchapter."})

                        st.rerun()  # Rerun to show success/initial message

                    except Exception as e:
                        st.error(f"Failed to start chat session: {e}")
                        reset_chat_state()
                        st.session_state.selected_subchapter_display_name = PLACEHOLDER_SELECT
                else:
                    st.error("Failed to initialize the LearnLM model after loading content.")
                    reset_chat_state()
                    st.session_state.selected_subchapter_display_name = PLACEHOLDER_SELECT
            else:
                # Content loading from S3 failed (error shown in load function)
                reset_chat_state()
                st.session_state.selected_subchapter_display_name = PLACEHOLDER_SELECT
        else:
            st.error(f"Internal error: Could not find object key for '{selected_display_name}'.")
            reset_chat_state()
            st.session_state.selected_subchapter_display_name = PLACEHOLDER_SELECT

# --- Display Chat History ---
st.markdown("---")
current_topic = st.session_state.selected_subchapter_display_name \
                if st.session_state.selected_subchapter_display_name != PLACEHOLDER_SELECT \
                else "No Subchapter Selected"
st.subheader(f"Chatting about: {current_topic}")

if st.session_state.selected_subchapter_display_name != PLACEHOLDER_SELECT and st.session_state.chat_session:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
elif st.session_state.selected_subchapter_display_name == PLACEHOLDER_SELECT and not st.session_state.messages:
    st.info("Select a subchapter from the dropdown menu above to start chatting.")

# --- Handle User Input ---
prompt_disabled = (st.session_state.selected_subchapter_display_name == PLACEHOLDER_SELECT or
                  not st.session_state.chat_session)

user_prompt = st.chat_input(
    f"Ask a question about {st.session_state.selected_subchapter_display_name}..." if not prompt_disabled else "Select a subchapter to enable chat",
    disabled=prompt_disabled,
    key="user_chat_input"
)

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    try:
        with st.spinner("Thinking..."):
            response = st.session_state.chat_session.send_message(user_prompt)
        assistant_response = response.text
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        # No need for chat_message context here, just append and rerun

        st.rerun()  # Rerun to display the new messages

    except Exception as e:
        st.error(f"An error occurred while communicating with LearnLM: {e}")
        error_message = f"Sorry, I encountered an error: {e}"
        st.session_state.messages.append({"role": "assistant", "content": error_message})
        st.rerun()  # Rerun to display error message