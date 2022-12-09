import numpy as np


def main():
    file1 = open('./LSTM_model_layer_normalization.csv', 'r')
    data1 = np.array(file1.readlines()[1:])
    # file2 = open('./CNN_model_full_training_data.csv', 'r')
    file2 = open('./CNN_model.csv', 'r')
    data2 = np.array(file2.readlines()[1:])
    
    new_data = []
    for i in range(len(data1)):
        prob1 = float(data1[i].split(',')[1])
        prob2 = float(data2[i].split(',')[1])
        if np.abs(prob1 - 0.5) > np.abs(prob2 - 0.5):
            new_data.append(data1[i])
        else:
            new_data.append(data2[i])
            
    output(new_data)
    
  
def output(probabilities):
    out_file = open('./combined.csv', 'w')
    out_file.write('ID,Predicted\n')
    for i, prob in enumerate(probabilities):
        out_file.write(prob)
    out_file.close()


main()