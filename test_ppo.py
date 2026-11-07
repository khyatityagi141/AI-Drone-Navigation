from stable_baselines3 import PPO
from drone_env import DroneEnv


# ==========================================
# LOAD ENVIRONMENT
# ==========================================

env = DroneEnv()


# ==========================================
# LOAD TRAINED PPO MODEL
# ==========================================

model = PPO.load("drone_ppo_model")


# ==========================================
# RESET ENVIRONMENT
# ==========================================

observation, info = env.reset()

total_reward = 0


print("\n================================")
print("STARTING TRAINED DRONE TEST")
print("================================\n")


# ==========================================
# RUN DRONE
# ==========================================

for step in range(500):

    # PPO chooses action
    action, _states = model.predict(
        observation,
        deterministic=True
    )

    # Apply action
    observation, reward, terminated, truncated, info = env.step(action)

    total_reward += reward

    print(
        f"Step: {step + 1:3d} | "
        f"Action: {action} | "
        f"Position: {env.drone} | "
        f"Reward: {reward:.2f}"
    )

    # Stop if episode ends
    if terminated or truncated:

        print("\nEpisode finished.")

        break


# ==========================================
# FINAL RESULT
# ==========================================

distance_to_target = (
    ((env.drone - env.target) ** 2).sum()
) ** 0.5


print("\n================================")
print("TEST COMPLETED")
print("================================")

print(f"Final Drone Position : {env.drone}")
print(f"Target Position      : {env.target}")
print(f"Distance to Target   : {distance_to_target:.2f}")
print(f"Total Reward         : {total_reward:.2f}")


if distance_to_target < 0.5:

    print("\nTARGET REACHED! 🚁🎯")

else:

    print("\nTARGET NOT REACHED.")