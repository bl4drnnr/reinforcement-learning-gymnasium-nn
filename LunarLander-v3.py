import os
from typing import Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm


class ActorCriticController:
    def __init__(self, environment, learning_rate: float, discount_factor: float) -> None:
        self.environment = environment
        self.discount_factor: float = discount_factor
        self.model: tf.keras.Model = self.create_actor_critic_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) # TODO: przygotuj odpowiedni optymizator, pamiętaj o learning rate!
        self.log_action_probability: Optional[tf.Tensor] = None  # zmienna pomocnicza, przyda się do obliczania docelowej straty
        self.tape: Optional[tf.GradientTape] = None  # zmienna pomocnicza, związana z działaniem frameworku
        self.last_error_squared: float = 0.0  # zmienna używana do wizualizacji wyników

    @staticmethod
    def create_actor_critic_model() -> tf.keras.Model:
        # TODO: przygotuj potrzebne warstwy sieci neuronowej o odpowiednich aktywacjach i rozmiarach
        input_layer = tf.keras.layers.Input(shape=(8,))  # 8 elementów stanu dla LunarLander

        x = tf.keras.layers.Dense(1024, activation='relu')(input_layer)
        x = tf.keras.layers.Dense(256, activation='relu')(x)

        actor_output = tf.keras.layers.Dense(4, activation='softmax')(x) # 4 akcje
        critic_output = tf.keras.layers.Dense(1, activation='linear')(x)

        return tf.keras.Model(inputs=input_layer, outputs=[actor_output, critic_output])

    def choose_action(self, state: np.ndarray) -> int:
        state = self.format_state(state)  # przygotowanie stanu do formatu akceptowanego przez framework

        self.tape = tf.GradientTape()
        with self.tape:
            # # wszystko co dzieje się w kontekście danej taśmy jest zapisywane i może posłużyć do późniejszego wyliczania pożądanych gradientów
            # action = None  # TODO: tu trzeba wybrać odpowiednią akcję korzystając z aktora
            # self.log_action_probability = None  # TODO: tu trzeba zapisać do późniejszego wykorzystania logarytm prawdopodobieństwa wybrania uprzednio wybranej akcji (będzie nam potrzebny by policzyć stratę aktora)
            policy, _ = self.model(state, training=True)  # ignorujemy wyjście krytyka
            distribution = tfp.distributions.Categorical(probs=policy)
            action = distribution.sample()
            self.log_action_probability = distribution.log_prob(action)
        return int(action)

    # noinspection PyTypeChecker
    def learn(self, state: np.ndarray, reward: float, new_state: np.ndarray, terminal: bool) -> None:
        state = self.format_state(state)
        new_state = self.format_state(new_state)

        with self.tape:  # to ta sama taśma, które użyliśmy już w fazie wybierania akcji
            # wszystko co dzieje się w kontekście danej taśmy jest zapisywane i może posłużyć do późniejszego wyliczania pożądanych gradientów

            # error = None  # TODO: tu trzeba obliczyć błąd wartościowania aktualnego krytyka
            # self.last_error_squared = float(error) ** 2

            # loss = None  # TODO: tu trzeba obliczyć sumę strat dla aktora i krytyka
            policy, value = self.model(state, training=True)
            _, next_value = self.model(new_state, training=True)

            value = tf.squeeze(value)
            next_value = tf.squeeze(next_value)

            if terminal:
                td_error = reward - value
            else:
                td_error = reward + self.discount_factor * next_value - value

            self.last_error_squared = float(td_error ** 2)

            critic_loss = td_error ** 2
            actor_loss = -td_error * self.log_action_probability

            total_loss = actor_loss + critic_loss

        gradients = self.tape.gradient(total_loss, self.model.trainable_weights)  # tu obliczamy gradienty po wagach z naszej straty, pomagają w tym informacje zapisane na taśmie
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))  # tutaj zmieniamy wagi modelu wykonując krok po gradiencie w kierunku minimalizacji straty

    @staticmethod
    def format_state(state: np.ndarray) -> np.ndarray:
        return np.reshape(state, (1, state.size))
    

def init_training(learning_rate: float, epochs: int, model_name: str):
    try:
        environment = gym.make(model_name)  # zamień na gym.make('LunarLander-v2', render_mode='human') by zająć się lądownikiem
        # zmień lub usuń render_mode, by przeprowadzić trening bez wizualizacji środowiska

        plots_path = f'plots/{model_name}_{learning_rate}'
        models_path = f'models/{model_name}_{learning_rate}'
        os.makedirs(plots_path, exist_ok=True)
        os.makedirs(models_path, exist_ok=True)

        controller = ActorCriticController(environment, learning_rate, 0.99)

        past_rewards = []
        past_errors = []
        for i_episode in tqdm(range(epochs)):  # tu decydujemy o liczbie epizodów
            done = False
            state, info = environment.reset()
            reward_sum = 0.0
            errors_history = []

            while not done:
                # environment.render()  # tą linijkę możemy wykomentować, jeżeli nie chcemy mieć wizualizacji na żywo

                action = controller.choose_action(state)
                new_state, reward, done, trunctated, info = environment.step(action)
                controller.learn(state, reward, new_state, done)
                state = new_state
                reward_sum += reward
                errors_history.append(controller.last_error_squared)

            past_rewards.append(reward_sum)
            past_errors.append(np.mean(errors_history))

            window_size = 50  # tutaj o rozmiarze okienka od średniej kroczącej
            if i_episode % 25 == 0:  # tutaj o częstotliwości zrzucania wykresów
                if len(past_rewards) >= window_size:
                    fig, axs = plt.subplots(2)
                    axs[0].plot(
                        [np.mean(past_errors[i:i + window_size]) for i in range(len(past_errors) - window_size)],
                        'tab:red',
                    )
                    axs[0].set_title('mean squared error')
                    axs[1].plot(
                        [np.mean(past_rewards[i:i+window_size]) for i in range(len(past_rewards) - window_size)],
                        'tab:green',
                    )
                    axs[1].set_title('sum of rewards')
                
                plot_path = f'plots/{model_name}_{learning_rate}/learning_{i_episode}.png'
                plt.savefig(plot_path)
                plt.clf()

                model_path = f'models/{model_name}_{learning_rate}/model_{i_episode}.keras'
                controller.model.save(model_path)

        environment.close()

        final_model_path = f"models/{model_name}_{learning_rate}/final.keras"
        controller.model.save(final_model_path)  # tu zapisujemy model
    except Exception as e:
        print(e)
        return None


init_training(0.000005, 3000, 'LunarLander-v3')
