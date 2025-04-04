import streamlit as st
import google.generativeai as genai
import os
import boto3
from botocore.exceptions import ClientError
from pathlib import Path

# --- Konfiguration ---
st.set_page_config(
    page_title="Lernen mit LearnLM",
    page_icon="📚",
    layout="wide"
)

# --- Geheimnisverwaltung & Client-Initialisierung ---
try:
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
    aws_access_key_id = st.secrets["AWS_ACCESS_KEY_ID"]
    aws_secret_access_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
    s3_bucket_name = st.secrets["S3_BUCKET_NAME"]
    aws_region = st.secrets.get("AWS_REGION", "us-east-1")  # Standardmäßig us-east-1, falls nicht angegeben

    # Vorhandensein prüfen
    if not gemini_api_key or not aws_access_key_id or not aws_secret_access_key or not s3_bucket_name:
        st.error("Eines oder mehrere erforderliche Geheimnisse fehlen.")
        st.stop()

except KeyError as e:
    st.error(f"Fehlendes erforderliches Streamlit-Geheimnis: {e}. Bitte konfigurieren Sie die Geheimnisse.")
    st.stop()
except Exception as e:
    st.error(f"Ein Fehler ist beim Laden der Geheimnisse aufgetreten: {e}")
    st.stop()

# Gemini konfigurieren
try:
    genai.configure(api_key=gemini_api_key)
except Exception as e:
    st.error(f"Fehler bei der Konfiguration von Google Generative AI: {e}")
    st.stop()

# --- AWS S3 Client-Initialisierung (Zwischengespeichert) ---
@st.cache_resource(show_spinner="Verbinde mit AWS S3...")
def get_s3_client():
    """Initialisiert und gibt einen S3-Client mit Anmeldeinformationen aus den Geheimnissen zurück."""
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region
        )

        # Verbindung testen, indem versucht wird, den Bucket abzurufen
        s3_client.head_bucket(Bucket=s3_bucket_name)
        print("Erfolgreich mit S3 verbunden.")
        return s3_client
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        if error_code == '404':
            st.error(f"S3-Bucket '{s3_bucket_name}' nicht gefunden. Überprüfen Sie den Bucket-Namen.")
        elif error_code == '403':
            st.error(f"Zugriff auf S3-Bucket '{s3_bucket_name}' verweigert. Überprüfen Sie die IAM-Berechtigungen.")
        else:
            st.error(f"Fehler beim Verbinden mit S3: {e}")
        return None
    except Exception as e:
        st.error(f"Fehler bei der Initialisierung des S3-Clients: {e}")
        return None

s3_client = get_s3_client()
if not s3_client:
    st.stop()  # Anhalten, wenn die Client-Initialisierung fehlgeschlagen ist

# LearnLM Modellkonfiguration
generation_config = {
    "temperature": 0.4,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

PLATZHALTER_AUSWAHL = "-- Unterkapitel auswählen --"

# --- S3 Hilfsfunktionen (Zwischengespeichert) ---

@st.cache_data(show_spinner="Verfügbare Unterkapitel auflisten...")
def get_available_subchapters_from_s3(bucket_name: str, _client) -> dict[str, str]:
    """
    Listet Objekte im S3-Bucket auf, analysiert Namen, die dem Muster entsprechen,
    und gibt ein Dictionary zurück, das {Anzeigename: Objektschlüssel} zuordnet.
    """
    subchapter_map = {}
    try:
        # Alle Objekte im Bucket auflisten
        response = _client.list_objects_v2(Bucket=bucket_name)

        if 'Contents' in response:
            for obj in response['Contents']:
                object_key = obj['Key']
                # Umgang mit potenzieller Ordnerstruktur im Objektschlüssel
                filename = Path(object_key).name
                stem = Path(filename).stem

                parts = stem.split('_')
                if len(parts) == 3 and filename.endswith(".txt"):
                    main_chapter_str, topic_str, subchapter_str = parts
                    display_name = subchapter_str
                    # Verwenden des vollständigen Objektschlüssels als Wert
                    subchapter_map[display_name] = object_key
                else:
                    print(f"Info: Überspringe Objekt mit unerwartetem Namensformat: {object_key}")

        # Umgang mit Paginierung, falls mehr als 1000 Objekte vorhanden sind
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
                        print(f"Info: Überspringe Objekt mit unerwartetem Namensformat: {object_key}")

    except ClientError as e:
        st.error(f"Fehler beim Auflisten von Dateien im S3-Bucket '{bucket_name}': {e}")
        return {}
    except Exception as e:
        st.error(f"Ein unerwarteter Fehler ist beim Auflisten von S3-Dateien aufgetreten: {e}")
        return {}

    sorted_subchapter_map = dict(sorted(subchapter_map.items()))
    return sorted_subchapter_map

@st.cache_data(show_spinner="Lade Inhalt des Unterkapitels...")
def load_subchapter_content_from_s3(bucket_name: str, object_key: str, _client) -> str | None:
    """Lädt den Inhalt eines bestimmten Objekts aus S3."""
    try:
        response = _client.get_object(Bucket=bucket_name, Key=object_key)
        content = response['Body'].read().decode('utf-8')
        return content
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        if error_code == 'NoSuchKey':
            st.error(f"Unterkapitel-Datei '{object_key}' nicht im S3-Bucket '{bucket_name}' gefunden.")
        elif error_code == 'AccessDenied':
            st.error(f"Zugriff auf Datei '{object_key}' verweigert. Überprüfen Sie die IAM-Berechtigungen.")
        else:
            st.error(f"Fehler beim Zugriff auf die S3-Datei '{object_key}': {e}")
        return None
    except Exception as e:
        st.error(f"Ein Fehler ist beim Lesen der Datei '{object_key}' aus S3 aufgetreten: {e}")
        return None

def initialize_learnlm_model(system_prompt: str) -> genai.GenerativeModel | None:
    """Initialisiert das GenerativeModel mit einer Systemanweisung."""
    try:
        model = genai.GenerativeModel(
            model_name="learnlm-1.5-pro-experimental",
            generation_config=generation_config,
            system_instruction=system_prompt,
        )
        return model
    except Exception as e:
        st.error(f"Fehler bei der Initialisierung des LearnLM-Modells: {e}")
        return None

# --- Reset-Funktion ---
def reset_chat_state():
    """Löscht den Chatverlauf und verwandte Sitzungszustandsvariablen."""
    st.session_state.messages = []
    st.session_state.learnlm_model = None
    st.session_state.chat_session = None
    st.session_state.subchapter_content = None
    print("Chat-Status zurückgesetzt.")

# --- Streamlit App UI und Logik ---

st.title("📚 Lernen mit LearnLM")
st.caption("Betrieben mit Google LearnLM 1.5 Pro Experimental")

# --- Zustandsinitialisierung ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_subchapter_display_name" not in st.session_state:
    st.session_state.selected_subchapter_display_name = PLATZHALTER_AUSWAHL
if "subchapter_content" not in st.session_state:
    st.session_state.subchapter_content = None
if "learnlm_model" not in st.session_state:
    st.session_state.learnlm_model = None
if "chat_session" not in st.session_state:
    st.session_state.chat_session = None
if "subchapter_map" not in st.session_state:
    # Karte mit der S3-Funktion und dem zwischengespeicherten Client laden
    st.session_state.subchapter_map = get_available_subchapters_from_s3(s3_bucket_name, s3_client)

# --- Unterkapitelauswahl ---
if not st.session_state.subchapter_map:
    st.warning(
        f"Keine gültigen Unterkapiteldateien im S3-Bucket '{s3_bucket_name}' gefunden oder das Auflisten ist fehlgeschlagen. "
        f"Stellen Sie sicher, dass Dateien vorhanden sind und das Format 'Haupt_Thema_Unterthema.txt' befolgen, und überprüfen Sie die Berechtigungen."
    )

# Optionen vorbereiten, einschließlich des Platzhalters
available_display_names = [PLATZHALTER_AUSWAHL] + list(st.session_state.subchapter_map.keys())

try:
    current_index = available_display_names.index(st.session_state.selected_subchapter_display_name)
except ValueError:
    current_index = 0

previous_selection = st.session_state.selected_subchapter_display_name

selected_display_name = st.selectbox(
    "Wählen Sie das Unterkapitel aus, das Sie lernen möchten:",
    options=available_display_names,
    key="subchapter_selector",
    index=current_index,
)

# --- Unterkapitel laden und Modell/Chat initialisieren ---
if selected_display_name != previous_selection:
    st.session_state.selected_subchapter_display_name = selected_display_name

    if selected_display_name == PLATZHALTER_AUSWAHL:
        reset_chat_state()
        st.info("Bitte wählen Sie ein Unterkapitel aus der Liste, um zu beginnen.")
        st.rerun()  # Sicherstellen, dass die UI nach dem Zurücksetzen vollständig aktualisiert wird

    else:
        st.info(f"Lade Unterkapitel: {selected_display_name} von S3...")
        reset_chat_state()

        # Den entsprechenden Objektschlüssel abrufen
        object_key_to_load = st.session_state.subchapter_map.get(selected_display_name)

        if object_key_to_load:
            # Inhalt mit der S3-Funktion und dem zwischengespeicherten Client laden
            content = load_subchapter_content_from_s3(s3_bucket_name, object_key_to_load, s3_client)

            if content:  # Prüfen, ob das Laden des Inhalts erfolgreich war
                st.session_state.subchapter_content = content

                # Den System-Prompt definieren
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

                    1. 📚 **Frag mich ab** – Teste mein Wissen
                    2. 💡 **Erkläre ein Konzept**
                    3. 🔄 **Verwende eine Analogie**
                    4. 🔍 **Vertiefe ein Thema**
                    5. 🧠 **Reflektiere oder fasse zusammen**
                    6. 🧩 **Erstelle eine Konzeptkarte**

                    Warte, bis sich der Lernende für einen Modus entscheidet.

                    Spezifisches Verhalten je nach Modus:

                    - **📚 Frag mich ab**: Stelle 1 Frage pro Durchlauf, beginnend einfach, dann steigend. Bitte um Begründung der Antwort. Wenn korrekt: loben. Wenn falsch: behutsam zur richtigen Lösung führen. Nach 5 Fragen: Zusammenfassung oder Fortsetzung anbieten. Verwende relevante Seitenangaben bei Bedarf (z. B. „Diese Info findest du auf [seite: 221]“).

                    - **💡 Erkläre ein Konzept**: Frage zuerst, welches Konzept erklärt werden soll. Gib eine schrittweise Erklärung. Biete relevante Seitenangaben zum Nachlesen an.

                    - **🔄 Verwende eine Analogie**: Wähle eine geeignete Stelle im Text aus und erkläre sie mithilfe eines kreativen, aber passenden Vergleichs. Nutze Seitenangaben zur Orientierung.

                    - **🔍 Vertiefe ein Thema**: Wenn der Lernende tiefer verstehen möchte, stelle offene, leitende Fragen. Nutze Seitenangaben zur Vertiefung.

                    - **🧠 Reflektiere oder fasse zusammen**: Fasse in eigenen Worten zusammen, was besprochen wurde. Stelle Reflexionsfragen wie: „Was fiel dir leicht? Wo möchtest du noch mehr üben?“ Gib ggf. Hinweise auf Seiten zum Wiederholen.

                    - **🧩 Konzeptkarte erstellen**: Bitte den Lernenden, 3–5 zentrale Ideen aus dem Kapitel zu nennen. Hilf, Zusammenhänge zu erkennen. Nutze Seitenzahlen zur Verankerung im Text.

                    Stil: Sei stets freundlich, unterstützend und geduldig. Stelle pro Antwort nur eine Frage oder Information. Fördere ein Gefühl von Fortschritt und Selbstwirksamkeit.

                    Bereit, mit dem Kapitel '{selected_display_name}' aus dem Lehrmittel Allgemeinbildung zu starten? Bitte den Lernenden, einen der 6 Lernmodi auszuwählen.
                    """


                # Das Modell initialisieren
                st.session_state.learnlm_model = initialize_learnlm_model(system_prompt)

                if st.session_state.learnlm_model:
                    # Chat-Sitzung starten
                    try:
                        st.session_state.chat_session = st.session_state.learnlm_model.start_chat(history=[])
                        st.success(f"Unterkapitel '{selected_display_name}' von S3 geladen. Fragen Sie mich alles dazu!")
                        # Optionale anfängliche Begrüßung
                        try:
                            initial_user_message = f"Bitte stell dich vor..."
                            initial_response = st.session_state.chat_session.send_message(initial_user_message)
                            st.session_state.messages.append({"role": "assistant", "content": initial_response.text})
                        except Exception as e:
                            st.warning(f"Konnte keine anfängliche Begrüßung von LearnLM erhalten: {e}")
                            st.session_state.messages.append({"role": "assistant", "content": f"Hallo! Ich bin bereit, Ihnen beim Verständnis der Konzepte in '{selected_display_name}' zu helfen. Fragen Sie mich gerne alles zu diesem Unterkapitel."})

                        st.rerun()  # Erneut ausführen, um Erfolg/initiale Nachricht anzuzeigen

                    except Exception as e:
                        st.error(f"Fehler beim Starten der Chat-Sitzung: {e}")
                        reset_chat_state()
                        st.session_state.selected_subchapter_display_name = PLATZHALTER_AUSWAHL
                else:
                    st.error("Fehler bei der Initialisierung des LearnLM-Modells nach dem Laden des Inhalts.")
                    reset_chat_state()
                    st.session_state.selected_subchapter_display_name = PLATZHALTER_AUSWAHL
            else:
                # Laden des Inhalts von S3 fehlgeschlagen (Fehler in der Ladefunktion angezeigt)
                reset_chat_state()
                st.session_state.selected_subchapter_display_name = PLATZHALTER_AUSWAHL
        else:
            st.error(f"Interner Fehler: Konnte keinen Objektschlüssel für '{selected_display_name}' finden.")
            reset_chat_state()
            st.session_state.selected_subchapter_display_name = PLATZHALTER_AUSWAHL

# --- Chat-Verlauf anzeigen ---
st.markdown("---")
current_topic = st.session_state.selected_subchapter_display_name \
                    if st.session_state.selected_subchapter_display_name != PLATZHALTER_AUSWAHL \
                    else "Kein Unterkapitel ausgewählt"
st.subheader(f"Chat über: {current_topic}")

if st.session_state.selected_subchapter_display_name != PLATZHALTER_AUSWAHL and st.session_state.chat_session:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
elif st.session_state.selected_subchapter_display_name == PLATZHALTER_AUSWAHL and not st.session_state.messages:
    st.info("Wählen Sie ein Unterkapitel aus dem Dropdown-Menü oben, um den Chat zu starten.")

# --- Benutzereingabe verarbeiten ---
prompt_disabled = (st.session_state.selected_subchapter_display_name == PLATZHALTER_AUSWAHL or
                    not st.session_state.chat_session)

user_prompt = st.chat_input(
    f"Stellen Sie eine Frage zu {st.session_state.selected_subchapter_display_name}..." if not prompt_disabled else "Wählen Sie ein Unterkapitel, um den Chat zu aktivieren",
    disabled=prompt_disabled,
    key="user_chat_input"
)

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    try:
        with st.spinner("Denke nach..."):
            response = st.session_state.chat_session.send_message(user_prompt)
        assistant_response = response.text
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        # Hier ist kein chat_message-Kontext erforderlich, nur anhängen und erneut ausführen

        st.rerun()  # Erneut ausführen, um die neuen Nachrichten anzuzeigen

    except Exception as e:
        st.error(f"Ein Fehler ist bei der Kommunikation mit LearnLM aufgetreten: {e}")
        error_message = f"Entschuldigung, ich bin auf einen Fehler gestoßen: {e}"
        st.session_state.messages.append({"role": "assistant", "content": error_message})
        st.rerun()  # Erneut ausführen, um die Fehlermeldung anzuzeigen