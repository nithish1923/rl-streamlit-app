import streamlit as st
import numpy as np
from agent import QAgent
from env import SimpleWalkEnv

st.set_page_config(page_title="RL Walker", layout="wide")

st.title("🤖 Reinforcement Learning Walker")

# Initialize
if "agent" not in st.session_state:
    st.session_state.agent = QAgent(10, 2)
    st.session_state.env = SimpleWalkEnv(10)
    st.session_state.rewards = []

agent = st.session_state.agent
env = st.session_state.env

# Controls
episodes = st.slider("Episodes", 10, 500, 100)

if st.button("Start Training"):
    st.session_state.rewards = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0

        for _ in range(50):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)

            agent.update(state, action, reward, next_state)

            state = next_state
            total_reward += reward

            if done:
                break

        st.session_state.rewards.append(total_reward)

# Visualization
st.subheader("📈 Reward Progress")
if st.session_state.rewards:
    st.line_chart(st.session_state.rewards)

# Show Q-table
st.subheader("🧠 Q-Table")
st.write(agent.q_table)

# Simulation
st.subheader("🚶 Simulation")

state = env.reset()
positions = []

for _ in range(20):
    action = agent.choose_action(state)
    state, _, _ = env.step(action)
    positions.append(state)

st.write("Agent Path:", positions)
