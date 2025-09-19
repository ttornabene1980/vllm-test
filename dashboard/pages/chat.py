import streamlit as st

# --- Windows-related context (could be expanded with docs, KB, manuals, etc.)
WINDOWS_CONTEXT = """
Windows is an operating system developed by Microsoft. 
It provides features such as the Start menu, File Explorer, Task Manager, 
registry settings, PowerShell, and supports multitasking and user accounts. 
Common issues include driver updates, security patches, and performance tuning.
"""

# --- Basic memory for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("💻 Windows Chat Assistant")

# --- Sidebar shows chat history
st.sidebar.header("Chat History")
for i, msg in enumerate(st.session_state.messages):
    role, content = msg["role"], msg["content"]
    st.sidebar.write(f"**{role.capitalize()} {i+1}:** {content}")

# --- Chat input
user_input = st.chat_input("Ask me something about Windows...")

if user_input:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Very basic response using "training context"
    # (replace with a real LLM call if you want)
    if "install" in user_input.lower():
        response = "To install software on Windows, you usually run the installer (.exe or .msi). You can also use `winget` or the Microsoft Store."
    elif "task manager" in user_input.lower():
        response = "You can open Task Manager by pressing Ctrl + Shift + Esc, or right-clicking the taskbar."
    else:
        response = f"Based on my Windows knowledge: {WINDOWS_CONTEXT[:150]}..."

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})

# --- Display current conversation
st.subheader("Conversation")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
