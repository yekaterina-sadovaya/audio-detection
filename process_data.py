import numpy as np
import os
import librosa
from scipy.io import wavfile


def add_rand_noise(data, noise_factor):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data


def manipulate(data, sampling_rate, pitch_factor):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)


def load_dataset(dataset):    
    root = 'E:/data/' + dataset
    label_filename = root + '.csv'
    
    # Read labels
    labels = {}
    label_file = open(label_filename)
    label_file.readline()
    label_str = label_file.readlines()
    label_file.close()
    
    for label in label_str:
        label = label.split(',')
        labels[label[0]] = label[2].replace('\n','')
    
    X = []
    y = []
    # Read audio files
    print('Loading dataset: ' + dataset)
    files = os.listdir(root)
    for i, filename in enumerate(files):
    
        # Print progress
        progress = i/(len(files)-1)*100
        print(f'\r|{round(progress)*"#"}{round(100-progress)*"-"}|  {progress:.2f} %', end = '')
    
        if filename.endswith('.wav'):
            # Add label of the audio file to y_test y
            y.append(labels[filename.replace('.wav','')])
            
            samplerate, audio_data = wavfile.read(root + '/' + filename)

            # Augmenting
            audio_data = manipulate(audio_data.astype(float), samplerate, np.random.randn())

            # Make audio files the same lenght and scale
            audio_data = audio_data[:441000]
            audio_data = np.append(audio_data, np.zeros(441000 - len(audio_data)))
            audio_data = audio_data * 1/np.max(np.abs(audio_data))
            
            # Extract Mel bands
            kwargs_for_mel = {'n_mels': 64}
            x = librosa.feature.melspectrogram(
                y=audio_data, 
                sr=samplerate, 
                n_fft=1024, 
                hop_length=512, 
                **kwargs_for_mel)
            X.append(x)
    print('\nDataset ' + dataset + ' loaded')
    return X, y
    
    
def load_dataset_test():    
    root = 'E:/data/test'
    
    X = []
    ids = []
    # Read audio files
    print('Loading test dataset')
    files = os.listdir(root)
    for i, filename in enumerate(files):
    
        # Print progress
        progress = i/(len(files)-1)*100
        #print(f'\r|{round(progress)*"#"}{round(100-progress)*"-"}|  {progress:.2f} %', end = '')
    
        if filename.endswith('.npy'):
            # Add name of audio to ids
            ids.append(filename.replace('.npy',''))
        
            audio_data = np.load(root + '/' + filename)
            
            # Scale audio files
            audio_data = audio_data * 1/np.abs(np.max(audio_data))
            
            # Extract Mel bands (using larger FFT window and hop lenght to have the same size output as with training data)
            kwargs_for_mel = {'n_mels': 64}
            x = librosa.feature.melspectrogram(
                y=audio_data, 
                sr=48000, 
                n_fft=1144, 
                hop_length=557, 
                **kwargs_for_mel)
            X.append(x)
    print('\nTest dataset loaded')
    return X, ids


def main():
    # X_1, y_1 = load_dataset('ff1010bird')
    # np.save('./data/X_1.npy', X_1)
    # np.save('./data/y_1.npy', y_1)
    # X_2, y_2 = load_dataset('warblrb10k_public')
    # np.save('./data/X_2.npy', X_2)
    # np.save('./data/y_2.npy', y_2)
    X_test, ids = load_dataset_test()
    np.save('./data/X_test.npy', X_test)
    np.save('./data/test_ids.npy', ids)
    print('All data was saved')


main()
