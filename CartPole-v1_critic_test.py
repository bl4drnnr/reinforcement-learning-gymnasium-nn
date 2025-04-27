import numpy as np
import tensorflow as tf

critic_model_0001_path = "./models/CartPole-v1_0.0001_critic_author/model_1625_critic.keras"
critic_model_00001_path = "./models/CartPole-v1_1e-05_critic_author/final_critic.keras"

critic_model_0001 = tf.keras.models.load_model(critic_model_0001_path)
critic_model_00001 = tf.keras.models.load_model(critic_model_00001_path)

states = {
    "Idealny stan": np.array([[0.0, 0.0, 0.0, 0.0]]),
    "Przechylony kijek": np.array([[0.0, 0.0, 0.5, 1.0]]),
    "Wagonik przy krawędzi": np.array([[2.0, 0.0, 0.0, 0.0]])
}

for name, state in states.items():
    value = critic_model_0001(state, training=False).numpy()[0][0]
    print(f"Critic model (learning_rate=0.0001, 1625 episodes): {name}: {value:.4f}")

for name, state in states.items():
    value = critic_model_00001(state, training=False).numpy()[0][0]
    print(f"Critic model (learning_rate=0.00001, final, 2000 episodes): {name}: {value:.4f}")
