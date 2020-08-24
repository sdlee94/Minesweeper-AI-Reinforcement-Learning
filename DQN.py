from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten
from keras.optimizers import Adam

def create_dqn(learn_rate, input_dims, n_actions):
    model = Sequential([
                Conv2D(128, (3,3), activation='relu', padding='same', input_shape=input_dims),
                Conv2D(128, (3,3), activation='relu', padding='same'),
                Conv2D(128, (3,3), activation='relu', padding='same'),
                Conv2D(128, (3,3), activation='relu', padding='same'),
                Flatten(),
                Dense(64, activation='relu'),
                Dense(n_actions, activation='linear')])

    model.compile(optimizer=Adam(lr=learn_rate), loss='mse')

    return model
