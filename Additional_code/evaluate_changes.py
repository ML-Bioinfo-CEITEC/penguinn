# Script for randomising input subsequences
# And extracting the "most important" subsequences for each input
# By degree of prediction score change
# Outputs a csv file with input id, starting position of subsequence and prediction change score

from tensorflow import keras
import numpy as np
import pandas as pd

PATH_TO_MODEL = "model.h5"

# Expects format of
# raw data string, model prediction
# column names "raw", "prediction"
PATH_TO_DATA = "raw_expected_predictions.csv"


def sequence_to_ohe(
        sequence,
        channel={
            'A': 0,
            'T': 1,
            'U': 1,
            'C': 2,
            'G': 3
        }
):
    # Transforms string into int array input for model
    sequence_size = len(sequence)
    ohe_dataset = np.zeros((sequence_size, 4))

    for pos, nucleotide in enumerate(sequence):
        if nucleotide == 'N':
            continue
        ohe_dataset[pos, channel[nucleotide]] = 1
    return ohe_dataset


def get_mutations(raw, changes):
    # "Mutates" a string with randomised change stretches
    result = []
    positions = []
    for i in range(len(raw) - 40):
        for change in changes:
            c_raw = raw.copy()
            for j in range(40):
                c_raw[i + j] = change[j]
            result.append(c_raw)
            positions.append(i)
    return result, positions


if __name__ == '__main__':
    model = keras.models.load_model(PATH_TO_MODEL)
    print("Loaded model.")
    print(model.summary())

    changes = pd.read_csv(r'./random_changes.csv')['changes'].tolist()
    print("Number of change strings {}.".format(len(changes)))

    result_id = []
    result_position = []
    result_diff = []

    df = pd.read_csv(PATH_TO_DATA)
    df.head()

    print("Beginning evaluation. Logs progress after processing 100 rows.")
    for index, row in df.iterrows():
        mutated_inputs, positions = get_mutations(list(row['raw']), changes)
        transformed_changes = np.array(list(map(sequence_to_ohe, mutated_inputs)))
        changes_predictions = np.concatenate(model.predict(transformed_changes)).tolist()

        prediction_differences = list(map(lambda x: x - row['predicted'], changes_predictions))

        changes_processed_df = pd.DataFrame(list(zip(positions, prediction_differences)),
                                            columns=['position', 'diff'])

        # we are looking for highest average negative change in prediction
        sorted_df = changes_processed_df.groupby('position').mean().sort_values('diff', ascending=True)

        result_id.append(index)
        result_position.append(sorted_df.index[0])
        result_diff.append(sorted_df.iloc[0]['diff'])

        if index != 0 and index % 1000 == 0:
            print("{} finished. Doing partial backup.".format(index))
            pd.DataFrame(
                list(zip(result_id, result_position, result_diff)), columns=['id', 'pos', 'diff']
            ).to_csv(r'output_backup' + str(index) + r'.csv', index=False, header=True)

        elif index != 0 and index % 100 == 0:
            print("{} finished.".format(index))

    final_df = pd.DataFrame(list(zip(result_id, result_position, result_diff)), columns=['id', 'pos', 'diff'])

    final_df.to_csv(r'output_final.csv', index=False, header=True)

