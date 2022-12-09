import numpy as np
from keras.models import load_model
import CNN_learn_semisupervised as my_lib


def load_data():
    X_test = np.load('./data/X_test.npy')
    X_test = X_test.reshape(-1,64,862,1)
    # X_test = np.swapaxes(X_test, 1,2)
    ids = np.load('./data/y_test.npy')
    return X_test, ids


def predict(X_test):
    # load model
    # model = load_model('best_model – kopio (2).h5', custom_objects={'semisupervised_loss': my_lib.semisupervised_loss})
    model = load_model('CNN_model2.h5')
    model.summary()
    
    predictions = model.predict(X_test)
    return predictions[:,0]
    
  
def output(probabilities, ids):
    # out_file = open('./CNN_model_semisupervised2.csv', 'w')
    out_file = open('./CNN_model.csv', 'w')
    out_file.write('ID,Predicted\n')
    for i, prob in enumerate(probabilities):
        out_file.write(ids[i].replace('.npy','') + ',' + str(prob) + '\n')
    out_file.close()
    
    
def main():
    X, ids = load_data()
    probabilities = predict(X)
    output(probabilities, ids)


main()