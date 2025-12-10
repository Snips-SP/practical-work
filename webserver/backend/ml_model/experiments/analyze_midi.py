import pretty_midi
import music21
import os
import re
import glob
import statistics
import matplotlib.pyplot as plt

PROGRESSIONS = {
    'Simple': {
        'bpm': 100,
        'chords': ['Am', 'C', 'D', 'F'],
        'timings': [16, 16, 16, 16],
    },
    'Complex': {
        'bpm': 120,
        'chords': ['Cm7', 'Fm7', 'Dm7b5', 'G7#5', 'Cm7'],
        'timings': [16, 16, 8, 8, 16],
    },
    'Cinematic': {
        'bpm': 70,
        'chords': ['Am', 'Fmaj7', 'C', 'G', 'Dm', 'Am', 'E7', 'Am9'],
        'timings': [32, 32, 32, 32, 16, 16, 32, 64],
    },
    'Neo-Soul_LoFi': {
        'bpm': 85,
        'chords': ['D-Maj9', 'Cm7', 'Fm9', 'B-m7', 'Eb9'],
        'timings': [16, 16, 16, 8, 8],
    },
    'Classical': {
        'bpm': 90,
        'chords': ['Dm', 'G7', 'Cmaj7', 'Fmaj7', 'Bm7b5', 'E7', 'Am'],
        'timings': [8, 8, 8, 8, 8, 8, 16],
    },
    'Modulation': {
        'bpm': 110,
        'chords': ['C', 'Am', 'F', 'G', 'E7', 'A', 'F#m', 'E'],
        'timings': [16, 16, 16, 16, 16, 16, 16, 16],
    }
}

def get_chord_pitches(chord_name):
    try:
        c = music21.harmony.ChordSymbol(chord_name)
        return {p.pitchClass for p in c.pitches}
    except Exception as e:
        print(f'Warning: music21 could not parse {chord_name}: {e}')
        return set()


def parse_filename(filename):
    try:
        # Remove extension
        name = os.path.splitext(filename)[0]
        parts = name.split('_')

        # Logic to map filename strings to PROGRESSIONS keys
        context = 'Unknown'
        if 'Simple' in name:
            context = 'Simple'
        elif 'Complex' in name:
            context = 'Complex'
        elif 'Cinematic' in name:
            context = 'Cinematic'
        elif 'Neo-Soul_LoFi' in name:
            context = 'Neo-Soul_LoFi'
        elif 'Classical' in name:
            context = 'Classical'
        elif 'Modulation' in name:
            context = 'Modulation'

        # Regex to find T(number) and P(number)
        temp_match = re.search(r'T([\d\.]+)', name)
        topp_match = re.search(r'P([\d\.]+)', name)

        temperature = temp_match.group(1) if temp_match else 'N/A'
        top_p = topp_match.group(1) if topp_match else 'N/A'

        model_name = parts[0]

        return {
            'model': model_name,
            'context': context,
            'temperature': temperature,
            'top_p': top_p,
            'filename': filename
        }
    except Exception as e:
        print(f'Error parsing filename {filename}: {e}')
        return None


def analyze_track(midi_path, context_key):
    if context_key not in PROGRESSIONS:
        print(f'Skipping {midi_path}: Unknown context "{context_key}"')
        return None

    data = PROGRESSIONS[context_key]
    expected_chords = data['chords']
    timings = data['timings']
    bpm = data['bpm']

    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f'Error loading MIDI {midi_path}: {e}')
        return None

    seconds_per_step = 60 / (bpm * 4)
    current_time = 0.0

    total_duration_analyzed = 0.0
    correct_duration = 0.0

    for chord_name, duration_steps in zip(expected_chords, timings):
        start_time = current_time
        end_time = current_time + (duration_steps * seconds_per_step)

        allowed_pitches = get_chord_pitches(chord_name)

        for instrument in pm.instruments:
            if instrument.is_drum: continue

            for note in instrument.notes:
                # Calculate Intersection
                # We want to know: "How much of this note is inside the current chord window?"
                max_start = max(note.start, start_time)
                min_end = min(note.end, end_time)

                overlap_duration = min_end - max_start

                # If there is valid overlap (duration > 0)
                if overlap_duration > 0:
                    total_duration_analyzed += overlap_duration

                    # Check Pitch Adherence
                    if (note.pitch % 12) in allowed_pitches:
                        correct_duration += overlap_duration
                    else:
                        pass

        current_time = end_time

    # Avoid division by zero if the file is empty
    if total_duration_analyzed > 0:
        total_accuracy = (correct_duration / total_duration_analyzed) * 100
    else:
        total_accuracy = 0.0

    return total_accuracy


def print_statistics(results):
    unique_models = sorted(list(set(r.get('model', 'Unknown') for r in results)))
    n_models = len(unique_models)

    data_by_config = {}
    data_by_ctx = {}
    all_config_keys = set()

    for model in unique_models:
        model_results = [r for r in results if r.get('model') == model]

        # Group by Config (Overall)
        data_by_config[model] = {}
        config_groups = {}
        for r in model_results:
            key = f'Temp:{r["temperature"]}, Top p:{r["top_p"]}'
            all_config_keys.add(key)
            if key not in config_groups:
                config_groups[key] = []
            config_groups[key].append(r['accuracy'])

        for k, vals in config_groups.items():
            data_by_config[model][k] = statistics.mean(vals)

        # Group by Context, then Config
        data_by_ctx[model] = {}
        ctx_groups = {}
        for r in model_results:
            c = r['context']
            if c not in ctx_groups:
                ctx_groups[c] = []
            ctx_groups[c].append(r)

        for ctx, items in ctx_groups.items():
            data_by_ctx[model][ctx] = {}
            c_conf_groups = {}
            for r in items:
                key = f'Temp:{r["temperature"]}, Top p:{r["top_p"]}'
                if key not in c_conf_groups:
                    c_conf_groups[key] = []
                c_conf_groups[key].append(r['accuracy'])

            for k, vals in c_conf_groups.items():
                data_by_ctx[model][ctx][k] = statistics.mean(vals)

    # Create consistent colors for configurations
    sorted_all_configs = sorted(list(all_config_keys))
    colors = plt.cm.viridis([i / len(sorted_all_configs) for i in range(len(sorted_all_configs))])
    config_color_map = dict(zip(sorted_all_configs, colors))

    # 2 Rows, N Columns
    fig, axes = plt.subplots(2, n_models, figsize=(6 * n_models, 10), sharey=True)

    # Handle the case of a single model (matplotlib returns a 1D array or single Axes)
    if n_models == 1:
        axes = [[axes[0]], [axes[1]]]

    for idx, model in enumerate(unique_models):
        ax_top = axes[0][idx]
        model_config_data = data_by_config.get(model, {})

        m_configs = sorted(list(model_config_data.keys()))
        m_means = [model_config_data[c] for c in m_configs]
        m_colors = [config_color_map[c] for c in m_configs]

        ax_top.bar(m_configs, m_means, color=m_colors)
        ax_top.set_title(f'{model}: Overall Config Performance')
        ax_top.set_ylabel('Avg Accuracy (%)')
        ax_top.set_ylim(0, 100)
        ax_top.grid(axis='y', alpha=0.3)
        ax_top.set_xticks([])

        ax_bottom = axes[1][idx]
        model_ctx_data = data_by_ctx.get(model, {})
        m_contexts = sorted(list(model_ctx_data.keys()))

        x_indices = range(len(m_contexts))
        bar_width = 0.8 / len(sorted_all_configs)

        for i, config in enumerate(sorted_all_configs):
            heights = []
            for ctx in m_contexts:
                val = model_ctx_data.get(ctx, {}).get(config, 0)
                heights.append(val)

            positions = [x + (i * bar_width) - (len(sorted_all_configs) * bar_width / 2) + (bar_width / 2) for x in
                         x_indices]
            ax_bottom.bar(positions, heights, width=bar_width, label=config, color=config_color_map[config])

        ax_bottom.set_title(f'{model}: Performance by Context')
        ax_bottom.set_xticks(x_indices)
        ax_bottom.set_xticklabels(m_contexts, rotation=30, ha='right')
        ax_bottom.set_xlabel('Context')
        ax_bottom.set_ylabel('Avg Accuracy (%)')
        ax_bottom.set_ylim(0, 100)
        ax_bottom.grid(axis='y', alpha=0.3)

    # Global Legend
    handles, labels = axes[1][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=len(sorted_all_configs),
               title="Hyperparameters")

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plot_filename = f'../plots/ChordAdherenceAccuracies.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')


def batch_process(input_directory):
    results = []
    files = glob.glob(os.path.join(input_directory, '*.mid'))
    print(f'Found {len(files)} MIDI files. Starting analysis...')
    print('-' * 50)

    for file_path in files:
        filename = os.path.basename(file_path)

        # Parse Metadata
        meta = parse_filename(filename)
        # Check if context is known in our dictionary
        if not meta or meta['context'] not in PROGRESSIONS:
            print(f'Skipping {filename}: Context unknown or parse error.')
            continue

        # Analyze Music
        accuracy = analyze_track(file_path, meta['context'])

        if accuracy is not None:
            entry = {
                **meta,
                'accuracy': round(accuracy, 2)
            }
            results.append(entry)
            print(f'{filename[:40]}, Accuracy: {entry["accuracy"]}%')

    print_statistics(results)



if __name__ == '__main__':
    INPUT_DIR = r'C:\Users\mbrun\Documents\University_Branche\Project\Documents\Final Tracks\Test Generations 1\midi'
    OUTPUT_FILE = 'generation_analysis_results.csv'

    batch_process(INPUT_DIR, OUTPUT_FILE)