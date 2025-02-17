import torch

from helper import chord2tokens
from encode import EncodingParameters

EncodingParameters.initialize()


def generate_from_chords(chords: list, timings: list):
    tokens = [chord2tokens(chord) for chord in chords]

    token_sequence = []
    for token, duration in zip(tokens, timings):
        # Repeat each chord for as many quarter notes as stated in timings
        for _ in range(duration):
            # Add whole chord to sequence followed by time note to differentiate between two different timings
            token_sequence.extend(token)
            token_sequence.append(EncodingParameters.time_note)

    model = torch.load('gpt_model_empty.ph')




generate_from_chords(['F', 'D', 'B'], [4, 8, 4])
