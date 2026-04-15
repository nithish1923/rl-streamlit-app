import streamlit as st
import numpy as np
import time

st.set_page_config(page_title="RL Bot Walker", layout="centered")

st.title("🤖 RL Walking Bot (Visual Demo)")

# Environment
SIZE = 10

# Q-learning setup
if "q_table" not in st.session_state:
    st.session_state.q_table = np.zeros((SIZE, 2))
    st.session_state.epsilon = 0.3
    st.session_state.lr = 0.1
    st.session_state.gamma = 0.9

q_table = st.session_state.q_table

def choose_action(state):
    if np.random.rand() < st.session_state.epsilon:
        return np.random.randint(2)
    return np.argmax(q_table[state])

def update(state, action, reward, next_state):
    best_next = np.max(q_table[next_state])
    q_table[state, action] += st.session_state.lr * (
        reward + st.session_state.gamma * best_next - q_table[state, action]
    )

# Controls
episodes = st.slider("Episodes", 10, 200, 50)

if st.button("🚀 Train & Watch Bot"):
    placeholder = st.empty()

    for ep in range(episodes):
        state = 0

        for step in range(20):
            action = choose_action(state)

            # Move
            if action == 1:
                next_state = min(SIZE - 1, state + 1)
            else:
                next_state = max(0, state - 1)

            reward = 1 if next_state == SIZE - 1 else -0.01

            update(state, action, reward, next_state)
            state = next_state

            # 🟢 VISUAL BOT
            line = ["⬜"] * SIZE
            line[state] = "🤖"

            placeholder.markdown(
                f"### Episode {ep+1}\n\n" + " ".join(line)
            )

            time.sleep(0.2)

            if state == SIZE - 1:
                break

st.subheader("🧠 Q-Table")
st.write(q_table)
