import streamlit as st
import google.generativeai as genai
import os
import json # To parse credentials
from pathlib import Path # Still useful for parsing filenames conceptually

# Import Google Cloud libraries
from google.cloud import storage
from google.oauth2 import service_account
from google.api_core.exceptions import NotFound, Forbidden # For error handling

# --- Configuration ---

st.set_page_config(
    page_title="Subchapter Chatbot (GCS)",
    layout="wide"
)

# --- Secret Management & Client Initialization ---

# Load secrets securely
try:
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
    gcs_bucket_name = st.secrets["GCS_BUCKET_NAME"]
    gcs_credentials_json = st.secrets["gcs_service_account_json"]

    # Validate presence
    if not gemini_api_key or not gcs_bucket_name or not gcs_credentials_json:
        st.error("One or more required secrets (GEMINI_API_KEY, GCS_BUCKET_NAME, GCS_CREDENTIALS) are missing.")
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

# --- GCS Client Initialization (Cached) ---
@st.cache_resource(show_spinner="Connecting to Google Cloud Storage...")
def get_gcs_client():
    """Initializes and returns a GCS client using credentials from secrets."""
    try:
        credentials_info = json.loads(gcs_credentials_json)
        credentials = service_account.Credentials.from_service_account_info(credentials_info)
        storage_client = storage.Client(credentials=credentials)
        # Test connection by trying to get the bucket (optional but good practice)
        storage_client.get_bucket(gcs_bucket_name)
        print("Successfully connected to GCS.") # Logs for debugging
        return storage_client
    except json.JSONDecodeError:
        st.error("Failed to parse GCS credentials JSON. Check the format in secrets.")
        return None
    except (NotFound, Forbidden):
         st.error(f"Error accessing GCS Bucket '{gcs_bucket_name}'. Check bucket name and service account permissions (needs Storage Object Viewer).")
         return None
    except Exception as e:
        st.error(f"Error initializing GCS client: {e}")
        return None

storage_client = get_gcs_client()
if not storage_client:
    st.stop() # Stop if client initialization failed

# LearnLM Model Configuration
generation_config = {
    "temperature": 0.4,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

PLACEHOLDER_SELECT = "-- Select a Subchapter --"

# --- GCS Helper Functions (Cached) ---

@st.cache_data(show_spinner="Listing available subchapters...")
def get_available_subchapters_from_gcs(bucket_name: str, _client: storage.Client) -> dict[str, str]:
    """
    Lists blobs in the GCS bucket, parses names matching the pattern,
    and returns a dictionary mapping {display_name: blob_name}.
    Uses _client parameter to benefit from st.cache_data.
    """
    subchapter_map = {}
    try:
        blobs = _client.list_blobs(bucket_name) # Add prefix='textbooks/' if files are in a folder
        for blob in blobs:
            blob_name = blob.name
            # Handle potential folder structure in blob name
            filename = Path(blob_name).name
            stem = Path(filename).stem

            parts = stem.split('_')
            if len(parts) == 3 and filename.endswith(".txt"):
                main_chapter_str, topic_str, subchapter_str = parts
                display_name = subchapter_str
                # Use the full blob name (including path if any) as the value
                subchapter_map[display_name] = blob_name
            else:
                print(f"Info: Skipping blob with unexpected name format: {blob_name}")

    except (NotFound, Forbidden) as e:
         st.error(f"Error listing files in GCS Bucket '{bucket_name}'. Check permissions or bucket name. Details: {e}")
         return {} # Return empty on error
    except Exception as e:
        st.error(f"An unexpected error occurred listing GCS files: {e}")
        return {}

    sorted_subchapter_map = dict(sorted(subchapter_map.items()))
    return sorted_subchapter_map

@st.cache_data(show_spinner="Loading subchapter content...")
def load_subchapter_content_from_gcs(bucket_name: str, blob_name: str, _client: storage.Client) -> str | None:
    """Loads the content of a specific blob from GCS."""
    try:
        bucket = _client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        if blob.exists():
            content = blob.download_as_text(encoding="utf-8")
            return content
        else:
            st.error(f"Subchapter file '{blob_name}' not found in GCS bucket '{bucket_name}'.")
            return None
    except (NotFound, Forbidden) as e:
         st.error(f"Error accessing GCS file '{blob_name}'. Check permissions or bucket/file name. Details: {e}")
         return None
    except Exception as e:
        st.error(f"An error occurred reading file '{blob_name}' from GCS: {e}")
        return None

def initialize_learnlm_model(system_prompt: str) -> genai.GenerativeModel | None:
    """Initializes the GenerativeModel with a system instruction."""
    # (No changes needed in this function itself)
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

st.title("📚 Subchapter Exam Prep Chatbot (GCS)")
st.caption("Powered by Google LearnLM 1.5 Pro Experimental | Content from GCS")

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
    # Load map using the GCS function and cached client
    st.session_state.subchapter_map = get_available_subchapters_from_gcs(gcs_bucket_name, storage_client)


# --- Subchapter Selection ---
if not st.session_state.subchapter_map:
    st.warning(
        f"No valid subchapter files found in GCS bucket '{gcs_bucket_name}' or failed to list them. "
        f"Ensure files exist and follow the 'Main_Topic_Subtopic.txt' format, and check permissions."
    )
    # Don't stop here, maybe the listing failed temporarily, let user see error messages above.
    # st.stop() # Consider if stopping is better if the list is essential

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
        st.rerun() # Ensure UI updates fully after reset

    else:
        st.info(f"Loading subchapter: {selected_display_name} from GCS...")
        reset_chat_state()

        # Get the corresponding blob name (which might include path)
        blob_name_to_load = st.session_state.subchapter_map.get(selected_display_name)

        if blob_name_to_load:
            # Load content using the GCS function and cached client
            content = load_subchapter_content_from_gcs(gcs_bucket_name, blob_name_to_load, storage_client)

            if content: # Check if content loading was successful
                st.session_state.subchapter_content = content

                # Define the system prompt
                system_prompt = f"""You are an expert tutor...
                Your knowledge is STRICTLY LIMITED to the following text from subchapter '{selected_display_name}'.
                ...
                --- START OF SUBCHAPTER '{selected_display_name}' TEXT ---
                {st.session_state.subchapter_content}
                --- END OF SUBCHAPTER '{selected_display_name}' TEXT ---
                ...
                Begin the conversation by introducing yourself... ready to discuss subchapter '{selected_display_name}'.
                """

                # Initialize the model
                st.session_state.learnlm_model = initialize_learnlm_model(system_prompt)

                if st.session_state.learnlm_model:
                    # Start chat session
                    try:
                        st.session_state.chat_session = st.session_state.learnlm_model.start_chat(history=[])
                        st.success(f"Subchapter '{selected_display_name}' loaded from GCS. Ask me anything about it!")
                        # Optional initial greeting
                        try:
                             initial_user_message = f"Please introduce yourself..." # As before
                             initial_response = st.session_state.chat_session.send_message(initial_user_message)
                             st.session_state.messages.append({"role": "assistant", "content": initial_response.text})
                        except Exception as e:
                            st.warning(f"Could not get initial greeting from LearnLM: {e}")
                            st.session_state.messages.append({"role": "assistant", "content": f"Hello! I'm ready to help..."}) # As before

                        st.rerun() # Rerun to show success/initial message

                    except Exception as e:
                         st.error(f"Failed to start chat session: {e}")
                         reset_chat_state()
                         st.session_state.selected_subchapter_display_name = PLACEHOLDER_SELECT
                else:
                    st.error("Failed to initialize the LearnLM model after loading content.")
                    reset_chat_state()
                    st.session_state.selected_subchapter_display_name = PLACEHOLDER_SELECT
            else:
                # Content loading from GCS failed (error shown in load function)
                reset_chat_state()
                st.session_state.selected_subchapter_display_name = PLACEHOLDER_SELECT
        else:
            st.error(f"Internal error: Could not find blob name for '{selected_display_name}'.")
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

        st.rerun() # Rerun to display the new messages

    except Exception as e:
        st.error(f"An error occurred while communicating with LearnLM: {e}")
        error_message = f"Sorry, I encountered an error: {e}"
        st.session_state.messages.append({"role": "assistant", "content": error_message})
        st.rerun() # Rerun to display error message