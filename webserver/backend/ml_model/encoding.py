from typing import Dict, List, Tuple
import numpy as np

class EncodingConfig:
    """
    EncodingConfig holds the token layout and helper mappings for notes, drums, and microtiming tokens.
    Call EncodingConfig.initialize() once at startup to populate maps.
    """

    # Public config
    tokens: List[int] = []
    padding_token: int = None
    vocab_size: int = None

    # Pianoroll config
    # Every quarter note is split into 24 steps in each pianoroll
    pianoroll_resolution: int = 24
    # We mainly encode only sixteenth notes, except for drums where we encode the whole pianoroll resolution
    encoding_resolution: int = 4

    # Instrument config
    # The order of tracks in which they appear in the dataset
    midi_tracks: List[str] = ['Drums', 'Piano', 'Guitar', 'Bass', 'Strings']
    # The order we wish to encode the notes in (As string and as indices of midi_tracks)
    encoding_order: List[str] = ['Drums', 'Bass', 'Piano', 'Guitar', 'Strings']
    # [3, 1, 2, 4, 0]
    trc_idx: List[int] = None

    programs = {
        'Drums': 0,
        'Piano': 0,
        'Guitar': 32,
        'Bass': 35,
        'Strings': 49
    }

    # General pitch packing / offset settings
    note_size: int = 84  # how many pitch slots per melodic instrument
    note_offset: int = 24  # midi pitch corresponding to index 0 inside each melodic instrument block

    # Special tokens (filled during initialize)
    time_note: int = None
    end_note: int = None
    begin_note: int = None

    # Drum pitches used in the dataset (sorted)
    drum_pitches: List[int] = [27, 28, 32, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 49, 50,
                               51, 53, 54, 55, 56, 57, 59, 60, 61, 62, 63, 64, 65, 67, 68, 69, 70, 73, 75,
                               76, 77, 80, 81, 82, 83, 85, 87]

    # Microtiming definitions: (delta in steps)
    microtimings: List[int] = [-2, -1, 1, 2, 3]

    # Mappings filled at initializing
    # Intervals for Drums, Microtiming, Drum+Micro, Melodic, Bass, Guitar, String, Piano and Special.
    # interval[0] <= instrument tokens <= interval[1]
    instrument_intervals: Dict[str, Tuple[int, int]] = {}
    drum_pitch_to_token: Dict[int, int] = {}
    drum_token_to_pitch: Dict[int, int] = {}
    microtiming_delta_to_token: Dict[int, int] = {}
    microtiming_token_to_delta: Dict[int, int] = {}

    @classmethod
    def initialize(cls):
        if cls.tokens:
            return  # already initialized

        cls.tokens = []
        cls.trc_idx = [cls.midi_tracks.index(track) for track in cls.encoding_order]
        current_token = 0

        # Drum tokens
        drum_base = current_token
        for midi_pitch in cls.drum_pitches:
            cls.drum_pitch_to_token[midi_pitch] = current_token
            cls.drum_token_to_pitch[current_token] = midi_pitch
            current_token += 1
        cls.instrument_intervals['Drums'] = (drum_base, current_token-1)

        # Microtiming tokens
        microtiming_base = current_token
        for delta in cls.microtimings:
            cls.microtiming_delta_to_token[delta] = current_token
            cls.microtiming_token_to_delta[current_token] = delta
            current_token += 1
        cls.instrument_intervals['Microtimings'] = (microtiming_base, current_token-1)
        cls.instrument_intervals['Drum+Micro'] = (drum_base, current_token - 1)

        # Add melodic instrument blocks in the chosen order: 'Bass', 'Piano', 'Guitar', 'Strings'
        melodic_base = current_token
        for t in [t for t in cls.encoding_order if t != 'Drums']:
            instrument_base = current_token
            # just reserve the integer range; tokens will be 0..(vocab-1) anyway
            current_token += cls.note_size
            cls.instrument_intervals[t] = (instrument_base, current_token-1)
        cls.instrument_intervals['Melodic'] = (melodic_base, current_token-1)

        instrument_base = current_token
        # Special tokens: time_note, end_note, padding
        cls.time_note = current_token
        current_token += 1
        cls.begin_note = current_token
        current_token += 1
        cls.end_note = current_token
        current_token += 1
        cls.padding_token = cls.end_note
        cls.instrument_intervals['Special'] = (instrument_base, current_token-1)

        # Final vocabulary size and token list
        cls.vocab_size = current_token
        cls.tokens = list(range(cls.vocab_size))

    @staticmethod
    def reorder_current(cur_seq):
        """
        Reorder the sequence in one timestep so we have Drums, Bass, Piano, Guitar, and Strings, efficiently in numpy.
        Assumes instruments are encoded in blocks of size EncodingConfig.note_size.
        """
        # Convert the list to a NumPy array
        seq_arr = np.array(cur_seq)

        # Define the boundaries for clarity
        drum_interval = EncodingConfig.instrument_intervals['Drum+Micro']
        melodic_interval = EncodingConfig.instrument_intervals['Melodic']

        # Create boolean masks to identify token from drums and micro tokens
        drum_mask = (drum_interval[0] <= seq_arr) & (seq_arr <= drum_interval[1])
        melodic_mask = (melodic_interval[0] <= seq_arr) & (seq_arr <= melodic_interval[1])
        # The other mask can be inferred from the first two
        other_mask = ~ (melodic_mask | drum_mask)

        # Apply masks to get the sections
        melodic_notes = seq_arr[melodic_mask]
        drum_notes = seq_arr[drum_mask]
        other_tokens = seq_arr[other_mask]

        melodic_notes.sort()

        return np.concatenate((drum_notes, melodic_notes, other_tokens)).tolist()

    @classmethod
    def token_to_info(cls, token: int) -> Dict[str, any]:
        """
        Reverse maps a token to its instrument, pitch, and octave.
        Returns a dictionary with keys: 'instrument', 'pitch', 'octave', and optionally 'tag' or 'delta'.
        """
        # Ensure mappings exist
        if not cls.tokens:
            raise RuntimeError("EncodingConfig not initialized")

        # 1. Check Special Tokens
        s_start, s_end = cls.instrument_intervals['Special']
        if s_start <= token <= s_end:
            tag = "Time" if token == cls.time_note else \
                "Begin" if token == cls.begin_note else \
                    "End" if token == cls.end_note else "Unknown"
            return {'instrument': 'Special', 'pitch': None, 'octave': None, 'tag': tag}

        # 2. Check Drums
        d_start, d_end = cls.instrument_intervals['Drums']
        if d_start <= token <= d_end:
            pitch = cls.drum_token_to_pitch[token]
            # Standard MIDI Octave: C4 (60) is octave 4.
            octave = (pitch // 12) - 1
            return {'instrument': 'Drums', 'pitch': pitch, 'octave': octave}

        # 3. Check Microtimings (No pitch/octave, return delta)
        m_start, m_end = cls.instrument_intervals['Microtimings']
        if m_start <= token <= m_end:
            delta = cls.microtiming_token_to_delta[token]
            return {'instrument': 'Microtiming', 'pitch': None, 'octave': None, 'delta': delta}

        # 4. Check Melodic Instruments
        # We iterate through the specific instruments defined in the order (excluding drums)
        melodic_instruments = [t for t in cls.encoding_order if t != 'Drums']

        for instr in melodic_instruments:
            start, end = cls.instrument_intervals[instr]
            if start <= token <= end:
                # Math: token = start + (pitch - offset)
                # Therefore: pitch = (token - start) + offset
                relative_pos = token - start
                pitch = relative_pos + cls.note_offset
                octave = (pitch // 12) - 1
                return {'instrument': instr, 'pitch': pitch, 'octave': octave}

        return {'instrument': 'Unknown', 'pitch': None, 'octave': None}

EncodingConfig.initialize()
