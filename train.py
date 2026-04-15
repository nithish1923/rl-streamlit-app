import gymnasium as gym
from stable_baselines3 import PPO

def train():
    env = gym.make("Humanoid-v4")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
    )

    model.learn(total_timesteps=200_000)  # increase later
    model.save("humanoid_model")

    print("✅ Training complete. Model saved.")

if __name__ == "__main__":
    train()
