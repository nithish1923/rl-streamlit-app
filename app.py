import streamlit as st
import os

st.set_page_config(page_title="Humanoid RL Dashboard", layout="wide")

st.title("🤖 Humanoid Reinforcement Learning Dashboard")

st.markdown("Train a humanoid to walk using MuJoCo + PPO")

# Buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("🚀 Train Model"):
        st.info("Training started... Check terminal")
        os.system("python train.py")

with col2:
    if st.button("🎥 Generate Video"):
        st.info("Recording humanoid...")
        os.system("python record.py")

# Show output
st.subheader("📺 Latest Simulation")

if os.path.exists("humanoid.gif"):
    st.image("humanoid.gif")
else:
    st.warning("No video yet. Train and generate video first.")

st.subheader("📊 Notes")
st.write("""
- Initial behavior: random falling
- Later: standing → walking
- Increase timesteps for better results
""")
