from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from drone_env import DroneEnv


# ==========================================
# CREATE DRONE ENVIRONMENT
# ==========================================

env = DroneEnv()


# ==========================================
# CHECK ENVIRONMENT
# ==========================================

print("Checking Drone Environment...")

check_env(env, warn=True)

print("Environment check completed!")


# ==========================================
# CREATE PPO MODEL
# ==========================================

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01
)


# ==========================================
# START TRAINING
# ==========================================

print("\n================================")
print("STARTING PPO TRAINING")
print("================================\n")

model.learn(
    total_timesteps=100000
)


# ==========================================
# SAVE TRAINED MODEL
# ==========================================

model.save("drone_ppo_model")


print("\n================================")
print("PPO TRAINING COMPLETED")
print("================================")
print("Model saved as: drone_ppo_model")