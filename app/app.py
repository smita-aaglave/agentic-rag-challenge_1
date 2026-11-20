
import streamlit as st
import json

from backend import initialize_backend, ask_agent

st.set_page_config(
    page_title="SkillBridge | Agentic RAG",
    page_icon="🎓",
    layout="wide"
)

# ---------------- PAGE DESIGN ----------------

st.title("🎓 SkillBridge")
st.subheader("Agentic RAG — Student Academic Assistant")

st.markdown(
    "Ask questions about student performance, attendance, "
    "assessments, courses and academic policies."
)

st.divider()

# ---------------- BACKEND INITIALIZATION ----------------

@st.cache_resource(show_spinner=False)
def load_backend():
    return initialize_backend()


try:
    with st.spinner("Initializing Agentic RAG system..."):
        load_backend()

    st.success("Agentic RAG system is ready!")

except Exception as e:
    st.error("Backend initialization failed.")
    st.exception(e)
    st.stop()


# ---------------- SESSION STATE ----------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------- DISPLAY CHAT HISTORY ----------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        if isinstance(message["content"], dict):
            st.json(message["content"])
        else:
            st.markdown(message["content"])


# ---------------- CHAT INPUT ----------------

question = st.chat_input(
    "Ask about a student, attendance, progress or course..."
)

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Agent is analyzing your question..."):

            try:
                response = ask_agent(question)

                if isinstance(response, str):
                    try:
                        response = json.loads(response)
                    except json.JSONDecodeError:
                        pass

                if isinstance(response, dict):

                    final_answer = response.get("final_answer")

                    if final_answer:
                        st.markdown(str(final_answer))

                        with st.expander("View Agent Response Details"):
                            st.json(response)

                    else:
                        st.json(response)

                else:
                    st.markdown(str(response))

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })

            except Exception as e:
                st.error(f"Error while processing your question: {e}")


# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.title("🎓 SkillBridge")

    st.caption("Agentic RAG Student Assistant")

    st.divider()

    st.markdown("### What can I help with?")

    st.markdown("""
    - Student academic summary
    - Attendance analysis
    - Course progress
    - Assessment performance
    - Academic policy questions
    """)

    st.divider()

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("Powered by Agentic RAG · Gemini · Groq")
