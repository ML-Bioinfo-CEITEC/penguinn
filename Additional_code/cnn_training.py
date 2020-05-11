from tensorflow import keras as K
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve
from matplotlib.pyplot import figure


def create_architecture(
        sequence_size=200,
        channel=4
):
    """
  fun create the model based on sequence input
  size.

  parameters:
  sequence_size = length of the sequence in nt
  channel = each channel corresponds to a nucleotide.
  """
    model = K.models.Sequential()

    model.add(K.layers.Conv1D(
        filters=16,
        kernel_size=8,
        padding='same',
        data_format="channels_last",
        activation='relu',
        input_shape=(sequence_size, channel)))

    model.add(K.layers.BatchNormalization())
    model.add(K.layers.MaxPooling1D())
    model.add(K.layers.Dropout(0.3))

    model.add(K.layers.Conv1D(
        filters=8,
        kernel_size=8,
        padding='same',
        data_format="channels_last",
        activation='relu'))

    model.add(K.layers.BatchNormalization())
    model.add(K.layers.MaxPooling1D())
    model.add(K.layers.Dropout(0.3))

    model.add(K.layers.Conv1D(
        filters=4,
        kernel_size=8,
        padding='same',
        data_format="channels_last",
        activation='relu'))

    model.add(K.layers.BatchNormalization())
    model.add(K.layers.MaxPooling1D())
    model.add(K.layers.Dropout(0.3))

    model.add(K.layers.Conv1D(
        filters=3,
        kernel_size=8,
        padding='same',
        data_format="channels_last",
        activation='relu'))

    model.add(K.layers.BatchNormalization())
    model.add(K.layers.MaxPooling1D())
    model.add(K.layers.Dropout(0.3))

    model.add(K.layers.Flatten())

    model.add(K.layers.Dense(512, activation="relu"))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dropout(0.2))

    model.add(K.layers.Dense(1, activation='sigmoid'))

    model.summary()
    return model


def compile_network(model):
    optimizer = K.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def sequence_to_ohe(
        dataset,
        sequence_size=50,
        channel={
            'A': 0,
            'T': 1,
            'U': 1,
            'C': 2,
            'G': 3
        }
):
    """
  fun builds the one hot encoding numpy array of each
  sample sequence.

  paramenters:
  sequence_size = can corresponds to the
  length of the input sequences
  (if all the same) or an arbitrary number can be defined.
  channel = the coding of nucleotides.

  """

    samples_size = len(dataset)
    ohe_dataset = np.zeros((samples_size, sequence_size, len(set(channel.values()))))

    for index, sequence in enumerate(dataset):
        for pos, nucleotide in enumerate(sequence):
            if nucleotide == 'N':
                continue
            ohe_dataset[index, pos, channel[nucleotide]] = 1

    return ohe_dataset


def load_dataset(filename):
    """
    fun loads dataset from 'filename' and prepares it for training
    It separates pseudo-randomly third of the set,
    that can be saved for later evaluation of the model
    :return four sets, X means sequences, y labels

    """

    df = pd.read_csv(filename, sep='\t', names=['sequence', 'label'])

    sequence_df = sequence_to_ohe(
        dataset=df['sequence'].tolist(),
        sequence_size=200,
    )

    labels_df = np.array(list(map((lambda x: 1 if x == 'positive' else 0), list(df['label']))))

    if sequence_df.shape[0] == labels_df.shape[0]:
        print('dataset OK')
    else:
        print('sequence and label shapes are different, something went wrong...')

    print(
        'sequence_df samples',
        sequence_df.shape,
        'labels_df samples',
        labels_df.shape,
        sep='\t'
    )

    X_train, X_test, y_train, y_test = train_test_split(
        sequence_df, labels_df, test_size=0.33, random_state=1989)

    print(
        'X_train sequences',
        X_train.shape[0],
        'y_train labels',
        y_train.shape[0],
        sep='\t'
    )

    print(
        'X_test sequences',
        X_test.shape[0],
        'y_test labels',
        y_test.shape[0],
        sep='\t'
    )

    return X_train, X_test, y_train, y_test


def draw_precision_recall_curve(model, sequences, labels):
    probs = model.predict(sequences)
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    plt.plot(recall, precision, marker='.', label='My model')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower center")
    plt.title('Precision recall curve')
    plt.show()


def ROC_curve(probs, labels):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    auc_score = roc_auc_score(labels, probs)
    print("AUC score: ", auc_score)
    figure(num=None, figsize=(12, 9), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(fpr, tpr, 'blue', label='My model')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower center")
    plt.title('ROC curve')
    plt.show()


def plot_history(history):
    """
    fun plots history of the training of the model,
    accuracy and loss of the training and validation set

    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Accuracy')
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Loss')
    plt.legend()
    plt.show()


def main():
    X_train, X_test, y_train, y_test = load_dataset('Datasets/train_set_1_1.txt')
    model = create_architecture(sequence_size=200, channel=4)
    model = compile_network(model=model)

    history = model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=25,
        validation_split=0.2
    )

    plot_history(history)

    metrics = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    print('model evaluation on unknown dataset [loss, accuracy]:', metrics)
    draw_precision_recall_curve(model, X_test, y_test)

    model.save('model.h5')


main()
