import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/chat"

st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 SHL Assessment Recommender")
st.write("Enter a hiring requirement and get recommended SHL assessments.")

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.text_area(
    "Enter job requirement",
    placeholder="Example: We are hiring a Java backend developer with Spring Boot, SQL and REST APIs.",
    height=130
)

if st.button("Get Recommendations"):
    if not user_input.strip():
        st.warning("Please enter a job requirement.")
    else:
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        try:
            with st.spinner("Finding best SHL assessments..."):
                response = requests.post(
                    API_URL,
                    json={"messages": st.session_state.messages},
                    timeout=60
                )

            if response.status_code == 200:
                data = response.json()

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": data["reply"]
                })

                st.subheader("Assistant Reply")
                st.write(data["reply"])

                st.subheader("Recommendations")

                if data["recommendations"]:
                    for rec in data["recommendations"]:
                        with st.container(border=True):
                            st.subheader(rec["name"])
                            st.write(f"**Type:** {rec['test_type']}")
                            st.link_button("Open Assessment", rec["url"])
                else:
                    st.info("No recommendations yet. The assistant may need more details.")

            else:
                st.error(f"API Error: {response.status_code}")
                st.write(response.text)

        except requests.exceptions.ConnectionError:
            st.error("Backend is not running. Start FastAPI first on port 8000.")
        except Exception as e:
            st.error(f"Something went wrong: {e}")

st.sidebar.title("Conversation History")

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

for msg in st.session_state.messages:
    st.sidebar.write(f"**{msg['role'].title()}:** {msg['content']}")