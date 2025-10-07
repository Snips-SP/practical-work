import os
import glob
from tqdm import tqdm


def verify_encoded_files(dataset_path: str):
    """
    Scans a dataset directory to count and compare the number of .npz files
    against their corresponding .npz.tmp encoded versions.

    This helps identify if any files were skipped during the encoding process.
    """
    print(f'Starting scan in: {dataset_path}')

    # Construct the search pattern for the original pianoroll files
    # This matches the structure 'lpd_5_cleansed/*/*/*/*/*.npz'
    glob_pattern = os.path.join(dataset_path, 'lpd_5_cleansed', '*', '*', '*', '*', '*.npz')

    # Find all the original .npz files first
    original_npz_files = glob.glob(glob_pattern)

    if not original_npz_files:
        print(f'\nError: No .npz files found with the pattern:\n{glob_pattern}')
        print('Please ensure the dataset path is correct and points to the root directory containing lpd_5_cleansed.')
        return

    tmp_file_count = 0
    missing_tmp_files = []

    # Iterate through the found .npz files and check for their .tmp counterparts
    for npz_path in tqdm(original_npz_files, desc='Verifying encoded files'):
        expected_tmp_path = npz_path + '.tmp'
        if os.path.exists(expected_tmp_path):
            tmp_file_count += 1
        else:
            missing_tmp_files.append(npz_path)

    npz_file_count = len(original_npz_files)

    # --- Print the final report ---
    print('\n' + '=' * 20)
    print('  Verification Report')
    print('=' * 20)
    print(f'Total original pianoroll files (.npz): {npz_file_count}')
    print(f'Total encoded temporary files (.npz.tmp): {tmp_file_count}')
    print('-' * 20)

    if npz_file_count == tmp_file_count:
        print('All original .npz files have a corresponding .tmp file.')
    else:
        print(f'There are {len(missing_tmp_files)} file(s) missing a .tmp version.')

        # List the first 10 missing files as examples
        if missing_tmp_files:
            print('\nHere are some examples of unencoded files:')
            for i, missing_file in enumerate(missing_tmp_files[:10]):
                # Print the path relative to the dataset directory for cleaner output
                relative_path = os.path.relpath(missing_file, dataset_path)
                print(f'  - {relative_path}')
            if len(missing_tmp_files) > 10:
                print(f'  ... and {len(missing_tmp_files) - 10} more.')


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Change the working directory to the script's directory
    os.chdir(script_dir)
    verify_encoded_files('../lpd_5')