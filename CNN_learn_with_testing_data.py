import numpy as np
from sklearn.model_selection import train_test_split
import keras.backend as K
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.losses import BinaryCrossentropy, Reduction
from keras.callbacks import ModelCheckpoint


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
    # 4 layer
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(BatchNormalization())
    # 5 Layer
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    model.compile(optimizer='adam', loss=semisupervised_loss, metrics=['accuracy'])
    return model


def load_validation():
    X_train = np.load('./data/X_train.npy')
    X_train = X_train.swapaxes(1,2).reshape(-1,64,862,1)
    X_test = np.load('./data/X_test.npy')
    X_test = X_test.swapaxes(1,2).reshape(-1,64,862,1)
    y_train = np.load('./data/y_train.npy').astype(float)
    y_test = np.load('./data/y_test.npy').astype(float)
    return X_train, X_test, y_train, y_test
    

def load():
    X = np.load('./data/X_1.npy')
    X = X.swapaxes(1,2).reshape(-1,64,862,1)
    y = np.load('./data/y_1.npy')
    return X, y
    
    
def load_test_data():
    X_to_predict = np.load('./data/X_test.npy')
    X_to_predict = X_to_predict.reshape(-1,64,862,1)
    # Use 0.5 to identify unknown labels
    y_none = [0.5] * len(X_to_predict)
    return X_to_predict, y_none
    
    
def unsupervised_loss(y_true, y_pred):
    separability = K.abs(y_pred - 0.5)+0.5
    loss = -K.log(separability)
    return loss
    
    
def semisupervised_loss(y_true, y_pred):
    supervised_loss = BinaryCrossentropy(reduction=Reduction.NONE)
    # Label is 0.5 for unknown data
    mask = K.cast(K.equal(y_true, K.constant(0.5)), 'float32')
    # Calculate loss for unknown data
    loss1 = mask * unsupervised_loss(y_true, y_pred)
    # Calculate loss for known data
    loss2 = (1 - mask) * K.reshape(supervised_loss(y_true, y_pred), (-1,1))
    loss = K.mean(K.concatenate((loss1, loss2))) * 2
    return loss
    
    
def main():
    # load full training dataset
    X, y = load()
    X_to_predict, y_none = load_test_data()
    X = np.concatenate((X, X_to_predict))
    y = np.concatenate((y, y_none))
    
    # load dataset with validation data
    X_train, X_test, y_train, y_test = load_validation()
    X_to_predict, y_none = load_test_data()
    X_train = np.concatenate((X_train, X_to_predict))
    y_train = np.concatenate((y_train, y_none))
    
    # Save best model
    checkpoint = ModelCheckpoint('best_model.h5', monitor='loss', verbose=1, save_best_only=True, mode='auto', period=1)
    
    model = define_model()
    model.summary()
    
    model.fit(X, y, epochs=10, batch_size=16, validation_data=(X_to_predict, y_none), callbacks=[checkpoint])
    model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))
    
    # save model
    model.save('CNN_model_semisupervised.h5')


if __name__ == "__main__":
    main()
