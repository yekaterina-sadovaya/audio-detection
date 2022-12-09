import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import MaxPooling1D
from keras.layers import AveragePooling1D
from keras.layers import LayerNormalization
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint


# define lstm model
def define_model():
    model = Sequential()
    # Layer 1
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(862, 64)))
    model.add(LayerNormalization())
    model.add(AveragePooling1D(pool_size=2))
    model.add(Dropout(0.1))
    # Layer 2
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(LayerNormalization())
    model.add(AveragePooling1D(pool_size=2))
    model.add(Dropout(0.1))
    # Layer 3
    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(LayerNormalization())
    model.add(AveragePooling1D(pool_size=2))
    model.add(Dropout(0.1))
    # Layer 4
    model.add(Bidirectional(LSTM(512)))
    model.add(LayerNormalization())
    model.add(Dropout(0.1))
    # Layer 5
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    # Layer 6
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def load_validation():
    X_train = np.load('./data/X_train.npy')
    X_test = np.load('./data/X_test.npy')
    y_train = np.load('./data/y_train.npy')
    y_test = np.load('./data/y_test.npy')
    return X_train, X_test, y_train, y_test
    

def load():
    X = np.load('./data/X.npy')
    y = np.load('./data/y.npy')
    return X, y
    

def main():
    # load dataset
    X_train, X_test, y_train, y_test = load_validation()
    X, y = load()
    print('All data was loaded')
    
    # define model
    model = define_model()
    model.summary()
    print('Model was defined')
    
    # Save best model
    checkpoint = ModelCheckpoint('best_model.h5', monitor='loss', verbose=1, save_best_only=True, mode='auto', period=1)

    # Fit model with testing data
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    
    model.fit(X_train, y_train, epochs=1, batch_size=16, validation_data=(X_test, y_test))
    
    # Fit model with full training data
    model.fit(X, y, epochs=100, batch_size=16, callbacks=[checkpoint])
    
    # save model
    model.save('LSTM_model.h5')


main()
