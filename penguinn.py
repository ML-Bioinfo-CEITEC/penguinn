#!/usr/bin/env python

import argparse
import random
from itertools import repeat

import numpy as np
from Bio import SeqIO
from tensorflow import keras as k


def parse_input():
    """
    function for parsing input parameters
    :return: dictionary of parameters
    """
    parser = argparse.ArgumentParser(description='For every sequence in fasta file count probability of forming G4')
    parser.add_argument('--input', required=True, metavar='<input_fasta_filename>')
    parser.add_argument('--output', required=True, metavar='<output_filename>')
    parser.add_argument('--model', default="Models/model_1_1.h5", metavar='<model_name>')
    args = parser.parse_args()
    return vars(args)


def sequence_to_ohe(
        sequence,
        channel={
            'A': 0,
            'T': 1,
            'C': 2,
            'G': 3
        }
):
    """
  fun builds the one hot encoding numpy array for a sequence.
  :param sequence
  :param channel: the coding of nucleotides.
  """

    sequence_size = len(sequence)
    ohe_dataset = np.zeros((1, sequence_size, 4))
    for pos, nucleotide in enumerate(sequence):
        nucleotide = nucleotide.upper()
        if nucleotide == 'N':
            continue
        try:
            ohe_dataset[0, pos, channel[nucleotide]] = 1
        except KeyError:
            print()
            print("Unknown nucleotide", nucleotide)
            print("Counting as N.")
            continue
    return ohe_dataset


def write_score(output_file, seq_id=None, score=None, header=False):
    """
    fun writes information about sequence and it's score to the output_file
    :param output_file
    :param seq_id: sequence id from fasta file
    :param score
    :param header: if true, fun prints header to file
    """
    if header:
        output_file.write('Sequence ID\tScore\n')
    else:
        output_file.write(seq_id + '\t\t' + str(score) + '\n')


def prolong_sequence(sequence, size):
    """
    fun prolongs sequence with Ns to have desired length
    :param sequence: original sequence
    :param size: desired length of the sequence
    :return: new prolonged sequence
    """
    left_N = random.randint(0, size - len(sequence))
    right_N = size - len(sequence) - left_N
    output_sequence = ''.join(repeat('N', left_N)) + sequence + ''.join(repeat('N', right_N))
    return output_sequence


def predict_probs(fasta, model, output):
    """
    fun predicts probabilities of forming G4 for sequences in fasta file
    :param fasta: input fasta file with sequences
    :param model: Keras model used for predicting
    :param output: output file to write probabilities to
    """
    sequence_max_length = 200
    sequence_min_length = 20

    for sequence in SeqIO.parse(fasta, "fasta"):
        if len(sequence.seq) > sequence_max_length:
            print()
            print("Sequence", sequence.id, "is too long. Sequence needs to be shorter or equal to",
                  sequence_max_length)
            print("Skipping")
            continue
        elif len(sequence.seq) < sequence_max_length:
            if len(sequence.seq) < sequence_min_length:
                print()
                print("Sequence", sequence.id, "is too short. Sequence needs to be longer or equal to",
                      sequence_min_length)
                print("Skipping")
                continue
            else:
                sequence.seq = prolong_sequence(sequence.seq, sequence_max_length)

        prob = model.predict(sequence_to_ohe(sequence.seq))
        write_score(output, sequence.id, prob[0][0])


def main():
    arguments = parse_input()

    output = open(arguments["output"], 'w')

    try:
        model = k.models.load_model(arguments["model"])
    except (IOError, ImportError):
        print()
        print("Can't load the model", arguments["model"])
        return

    print("===========================================")

    try:
        fasta = open(arguments["input"], 'r')
    except OSError:
        print()
        print("Can't open file", arguments["input"])
        return

    write_score(output, header=True)
    predict_probs(fasta, model, output)


main()
