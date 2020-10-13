from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten
from keras.optimizers import Adam

def create_policy_network(learn_rate, input_dims, n_actions, conv_units, dense_units):
    model = Sequential([
                Conv2D(conv_units, (3,3), activation='relu', padding='same', input_shape=input_dims),
                Conv2D(conv_units, (3,3), activation='relu', padding='same'),
                Conv2D(conv_units, (3,3), activation='relu', padding='same'),
                Conv2D(conv_units, (3,3), activation='relu', padding='same'),
                Flatten(),
                Dense(dense_units, activation='relu'),
                Dense(dense_units, activation='relu'),
                Dense(n_actions, activation='softmax')])

    model.compile(optimizer=Adam(lr=learn_rate), loss='categorical_crossentropy')

    return model
