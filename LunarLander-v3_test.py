import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('./models/LunarLander-v3_5e-06/final.keras')

test_states = {
    "Stabilny przed ladowaniem": np.array([[0.0, 0.1, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0]]),
    "Duzy przechyl w lewo": np.array([[0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0]]),
    "Duzy przechyl w prawo": np.array([[0.0, 0.5, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0]]),
    "Mocne opadanie": np.array([[0.0, 0.05, 0.0, -0.8, 0.0, 0.0, 0.0, 0.0]])
}

for name, state in test_states.items():
    policy, value = model(state, training=False)
    policy = policy.numpy()[0]
    value = value.numpy()[0][0]
    action = np.argmax(policy)
    
    print(f"{name}:")
    print(f"\tWartosc stanu (ocena krytyka): {value:.4f}")
    print(f"\tProponowana akcja (aktora): {action} (rozklad: {policy})")
    print()
