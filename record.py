import gymnasium as gym
import imageio
from stable_baselines3 import PPO

def record():
    env = gym.make("Humanoid-v4", render_mode="rgb_array")
    model = PPO.load("humanoid_model")

    frames = []
    obs, _ = env.reset()

    for _ in range(500):
        action, _ = model.predict(obs)
        obs, _, done, _, _ = env.step(action)

        frame = env.render()
        frames.append(frame)

        if done:
            obs, _ = env.reset()

    imageio.mimsave("humanoid.gif", frames, fps=30)
    print("🎥 Video saved as humanoid.gif")

if __name__ == "__main__":
    record()
