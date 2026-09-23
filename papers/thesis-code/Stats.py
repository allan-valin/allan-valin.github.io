import os
import re
import time
from typing import Dict
from PathsAndConfigs import MANUAL_LABEL_READ_PATH, get_all_filepaths_from_directory
from src.DataClasses import ExampleData
from src.Model import check_with_model

# Initialize overall totals
overall_totals = {
    'total_labeled': 0,
    'total_predicted': 0,
    'error_quantity': 0,
    'exact_match_quantity': 0,
    'extra_peaks': 0,
    'fewer_peaks': 0,
    'peaks_over_1_2': 0,
    'max_peaks_per_window': 0
}
window_counter = 0
current_window_peaks = 1
start_time = time.time()

# Dictionary to hold totals for each WFM group
wfm_groups: Dict[str, Dict] = {}

print("Calculating...")
for example_filepath in get_all_filepaths_from_directory(MANUAL_LABEL_READ_PATH):
    main_example = ExampleData.load_from_file(example_filepath)
    
    # Extract the x_y part from the filename
    match = re.search(r"(\d+_\d+)\.npz$", example_filepath) # Filter 0_00
    #match = re.search(r"(\d+)_(\d+)_(\d+)\.npz$", example_filepath) # Filter 0000000_0_00

    if not match:
        continue  # Skip files that don't match the pattern

    # Filter 0_00
    x_y_part = match.group(1)
    wfm_group = x_y_part.split('_')[0]  # Extract WFM group (e.g., '0' from '0_123')

    # Filter 0000000_0_00
    # timestamp_part = match.group(1)
    # group_part = match.group(2)
    # subgroup_part = match.group(3)
    # wfm_group = f"{timestamp_part}_{group_part}"  # Combine timestamp and first number
    # x_y_part = f"{timestamp_part}_{group_part}_{subgroup_part}"  # Full identifier for display

    # Initialize WFM group BEFORE processing windows
    if wfm_group not in wfm_groups:
        wfm_groups[wfm_group] = {
            'total_labeled': 0,
            'total_predicted': 0,
            'error_quantity': 0,
            'exact_match_quantity': 0,
            'extra_peaks': 0,
            'fewer_peaks': 0,
            'peaks_over_1_2': 0  # Now initialized here first
        }
    
    # Initialize per-file counters
    file_total_labeled = 0
    file_total_predicted = 0
    file_error_quantity = 0
    file_exact_match_quantity = 0
    file_extra_peaks = 0
    file_fewer_peaks = 0
    file_peaks_over_1_2 = 0



    for child_example in main_example.all_windows_iter:
        # Get values from the model(?)
        predicted_peaks = check_with_model(child_example.y)
        amount_of_peaks = len(child_example.peaks)

        # Add to the total for printing
        file_total_labeled += amount_of_peaks
        file_total_predicted += predicted_peaks
        
        # Count peaks over 1.2
        # peaks_over_threshold = sum(1 for y in child_example.peaks_y if y <= -1.2)
        peaks_over_threshold = sum(1 for y in child_example.peaks_y if y <= -0.8)
        #peaks_over_threshold = sum(1 for y in child_example.peaks_y if y <= -0.25)
        overall_totals['peaks_over_1_2'] += peaks_over_threshold
        wfm_groups[wfm_group]['peaks_over_1_2'] += peaks_over_threshold
        file_peaks_over_1_2 += peaks_over_threshold

        # Shows peak number for each window
        current_window_peaks = amount_of_peaks
        # print(f"{current_window_peaks}")
        # print(f"Peaks in window # {window_counter} = {current_window_peaks}")
        if amount_of_peaks > overall_totals['max_peaks_per_window']:
            overall_totals['max_peaks_per_window'] = amount_of_peaks
        window_counter += 1

        # Checks for errors and sum them
        if amount_of_peaks < predicted_peaks: # Extra peaks
            diff = predicted_peaks - amount_of_peaks
            file_extra_peaks += diff
            file_error_quantity += diff
            current_window_peaks = 0
        elif amount_of_peaks > predicted_peaks: # Fewer Peaks
            diff = amount_of_peaks - predicted_peaks
            file_fewer_peaks += diff
            file_error_quantity += diff
            current_window_peaks = 0
        else: # Exact matches
            file_exact_match_quantity += amount_of_peaks
            current_window_peaks = 0
    
    # Update WFM group totals
    # wfm_groups[wfm_group]['total_labeled'] += file_total_labeled
    # wfm_groups[wfm_group]['total_predicted'] += file_total_predicted
    # wfm_groups[wfm_group]['error_quantity'] += file_error_quantity
    # wfm_groups[wfm_group]['exact_match_quantity'] += file_exact_match_quantity
    # wfm_groups[wfm_group]['extra_peaks'] += file_extra_peaks
    # wfm_groups[wfm_group]['fewer_peaks'] += file_fewer_peaks
    
    # Update overall totals
    overall_totals['total_labeled'] += file_total_labeled
    overall_totals['total_predicted'] += file_total_predicted
    overall_totals['error_quantity'] += file_error_quantity
    overall_totals['exact_match_quantity'] += file_exact_match_quantity
    overall_totals['extra_peaks'] += file_extra_peaks
    overall_totals['fewer_peaks'] += file_fewer_peaks
    
    # Print per-file statistics
    print(f"\nFile {x_y_part}:")
    print(f"Labeled Peaks = {file_total_labeled}")
    print(f"Predicted Peaks = {file_total_predicted}")
    print(f"Peak Difference = {abs(file_total_labeled - file_total_predicted)}")
    print(f"Total errors: {file_error_quantity}")
    print(f"Total of exact matches: {file_exact_match_quantity}")
    print(f"EXTRA: {file_extra_peaks}")
    print(f"FEWER: {file_fewer_peaks}")
    print(f"Peaks over 1.2: {file_peaks_over_1_2}")
    # print(f"Peaks over 0.8: {file_peaks_over_1_2}")
    # print(f"Peaks over 0.25: {file_peaks_over_1_2}")

# Print statistics for each WFM group
# for group in sorted(wfm_groups.keys()):
#     group_stats = wfm_groups[group]
    # print(f"\nWFM Group {group}:")
    # print(f"Labeled Peaks = {group_stats['total_labeled']}")
    # print(f"Predicted Peaks = {group_stats['total_predicted']}")
    # print(f"Peak Difference = {abs(group_stats['total_labeled'] - group_stats['total_predicted'])}")
    # print(f"Total errors: {group_stats['error_quantity']}")
    # print(f"Total of exact matches: {group_stats['exact_match_quantity']}")
    # print(f"EXTRA: {group_stats['extra_peaks']}")
    # print(f"FEWER: {group_stats['fewer_peaks']}")
    # print(f"Peaks over 1.2: {group_stats['peaks_over_1_2']}")
    # print(f"Peaks over 0.8: {group_stats['peaks_over_1_2']}")
    # print(f"Peaks over 0.25: {group_stats['peaks_over_1_2']}")

# Print overall statistics
print("\nOverall Totals Across All Files:")
# print(f"Labeled Peaks = {overall_totals['total_labeled']}")
# print(f"Predicted Peaks = {overall_totals['total_predicted']}")
# print(f"Peak Difference = {abs(overall_totals['total_labeled'] - overall_totals['total_predicted'])}")
# print(f"Total errors: {overall_totals['error_quantity']}")
# print(f"Total of exact matches: {overall_totals['exact_match_quantity']}")
# print(f"EXTRA: {overall_totals['extra_peaks']}")
# print(f"FEWER: {overall_totals['fewer_peaks']}")
# print(f"Peaks over 1.2: {overall_totals['peaks_over_1_2']}")
# print(f"Peaks over 0.8: {overall_totals['peaks_over_1_2']}")
# print(f"Peaks over 0.25: {overall_totals['peaks_over_1_2']}")

# Show only the values for easier copying
print(f"{overall_totals['total_labeled']}")
print(f"{overall_totals['total_predicted']}")
print(f"{abs(overall_totals['total_labeled'] - overall_totals['total_predicted'])}")
print(f"{overall_totals['error_quantity']}")
print(f"{overall_totals['exact_match_quantity']}")
print(f"{overall_totals['extra_peaks']}")
print(f"{overall_totals['fewer_peaks']}")
print(f"{overall_totals['peaks_over_1_2']}")
print("\nMaximum peaks in a single window:", overall_totals['max_peaks_per_window'])

end_time = time.time()

execution_time = end_time - start_time
print(f"Execution time: {execution_time:.6f} seconds")