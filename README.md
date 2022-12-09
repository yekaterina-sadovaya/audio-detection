# Bird Audio Detection

## This repository provides an implementation of CNN and LSTM models for bird audio detection

### Data
Three different datasets are given, two to be used for the optimization and one for the evaluation of the methods. The data can be downloaded from https://www.kaggle.com/competitions/bird-audio-detection/overview/datasets.
##### Training Data
Training datasets consist of 10 second long audio samples, and associated labels for each audio sample. The labels are in the form "0" or "1", where "1" means that there is a bird sound in the 10 second long audio sample.
##### Testing Data
The testing dataset is the "Bird Vocalisation Activity (BiVA) database: annotated soundscapes from the Chernobyl Exclusion Zone" [1], which contains data supplied by Natural Environment Research Council, and is available under the Open Government Licence v3.

### Competition Description

The task is to design a system that, given a short audio recording, returns a binary decision for the presence/absence of bird sound (bird sound of any kind).
