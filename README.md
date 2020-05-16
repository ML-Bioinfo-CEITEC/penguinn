# Penguinn

PENGUINN (Precise Exploration of Nuclear G-quadruplexes Using Interpretable Neural Networks) is a machine learning method based on Convolutional Neural Networks, that learns the characteristics of G4 sequences and predicts probability of forming G4s for given sequences.

##Installation

Using Git:

```
git clone https://gitlab.com/RBP_Bioinformatics/penguinn.git
```
or

```
git clone git@gitlab.com:RBP_Bioinformatics/penguinn.git
```

### Prerequisities

Penguinn is implemented in python using Keras and Tensorflow backend.

Required:

* python, recommended version 3.7
    * Keras 2.3.1
    * tensorflow 2.0.0
    * Biopython
    * numpy

### Running prediction

Follow the instructions:

```
cd path/to/Penguinn/directory
#add rights to execute
chmod +x penguinn.py
#run the prediction
./penguinn.py --input <input_fasta_file> --output <output_file> --model <path_to_model.h5>
```

Default model is set to model trained on dataset with positive:negative samples ratio 1:1, as described in our paper.

