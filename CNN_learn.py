import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers import Dropout


# define cnn model
def define_model():
    model = Sequential()
    # 1 layer
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(64, 862, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.1))
    # 2 layer
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    # 3 layer
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def load():
    X_1 = np.load('./data/X_1.npy')
    X_1 = X_1.reshape(X_1.shape[0],64,862,1)
    y_1 = np.load('./data/y_1.npy').astype(int)
    
    X_2 = np.load('./data/X_2.npy')
    X_2 = X_2.reshape(X_2.shape[0],64,862,1)
    y_2 = np.load('./data/y_2.npy').astype(int)
    
    X = np.concatenate((X_1, X_2))
    y = np.concatenate((y_1, y_2))
    return X, y
    

def main():
    # load dataset
    X, y = load()
    print('All data was loaded')
    
    # define model
    model = define_model()
    model.summary()
    print('Model was defined')

    # Fit model with testing data
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    # model.fit(X_train, y_train, epochs=20, batch_size=20, validation_data=(X_test, y_test))
    
    # Fit model with full training data
    model.fit(X, y, epochs=10, batch_size=20)
    
    # save model
    model.save('CNN_model2.h5')


main()
