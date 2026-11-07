from drone_env import DroneEnv

env = DroneEnv()

observation, info = env.reset()

print("================================")
print("   DRONE ENVIRONMENT TEST")
print("================================")

print("Initial Observation:")
print(observation)

for i in range(10):

    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)

    print()
    print("Step:", i + 1)
    print("Action:", action)
    print("Observation:", observation)
    print("Reward:", reward)

    if terminated or truncated:
        print("Episode finished.")
        break

env.close()

print()
print("================================")
print("   ENVIRONMENT TEST COMPLETE")
print("================================")