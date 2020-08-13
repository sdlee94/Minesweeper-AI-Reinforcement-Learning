from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten
from keras.optimizers import Adam

def create_dqn(learn_rate, input_dims, n_actions):
    model = Sequential([
                    Conv2D(64, (3,3), activation='relu', input_shape=input_dims),
                    Conv2D(64, (3,3), activation='relu'),
                    Conv2D(64, (3,3), activation='relu'),
                    Flatten(),
                    Dense(64, activation='relu'),
                    Dense(n_actions)])

    model.compile(optimizer=Adam(lr=learn_rate), loss='mse')

    return model
