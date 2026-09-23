import shutil
import subprocess
import importlib
import sys
import os
from tensorflow.python.ops.gen_logging_ops import Print

from TrainingDataGeneration import run_datagen
from TrainingExample import run_training
from ProcessRawWFM import process_wfm
from AutomaticLabeling import autolabel
from src import PathsAndConfigs, TrainingDataGeneration
from src.PathsAndConfigs import get_all_filepaths_from_directory, TRAINING_DATA_INPUT_PATH, VAL_DATA_INPUT_PATH, \
    CONVERTED_WFM_PATH, MANUAL_LABEL_READ_PATH, copy_all_files_from_dir_to_dir
from TrainingUtils import AMOUNT_OF_CLASSES

# at the last line of TrainingDataGeneration and TrainingExample, comment it out before running this

# This should be the equivalence between WINDOW_SIZE and max_peak
###################################
# Windows | WINDOW_SIZE | max_peak
# 16	    78125	        81
# 32	    39062.5	        41
# 64	    19531.25	    20
# 128	    9765.63	        10
# 256	    4882.81	        5
# 512	    2441.41	        3
# 1024	    1220.70	        1
# 2048	    610.35	        1
# 4096	    305.18	        1
###################################

RED = '\033[91m'
GREEN = '\033[92m'
BLUE = '\033[94m'
YELLOW = '\033[93m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
WHITE = '\033[97m'
RESET = '\033[0m'
# print(f"{RED}Error: File not found{RESET}")
# print(f"{GREEN}Success: Data saved{RESET}")
# print(f"{BLUE}Info: Processing...{RESET}")

# Default values from earlier tests with manual labels
window_size = 12500
max_peaks = int(13)
window_amount = 100
pulse_width_ns = 15
height = 0.2

# Define the path for the text file to store the values
config_file = "default_values.txt"


###################
# File Management #
###################


# Reads values from the text file and updates global variables
def read_and_apply_values_from_file():
    global window_size, max_peaks, window_amount, pulse_width_ns, height

    if os.path.exists(config_file):
        with open(config_file, "r") as file:
            for line in file:
                key, value = line.strip().split("=")
                if key == "window_size":
                    window_size = int(value)
                elif key == "max_peaks":
                    max_peaks = int(value)
                elif key == "window_amount":
                    window_amount = int(value)
                elif key == "pulse_width_ns":
                    pulse_width_ns = int(value)
                elif key == "height":
                    height = float(value)
    else:
        print(f"{config_file} not found.")
        # write_values_to_file()  # Create the file with default values

# Writes the current variable values to a text file
def write_values_to_file():
    with open(config_file, "w") as file:
        file.write(f"window_size={window_size}\n")
        file.write(f"max_peaks={max_peaks}\n")
        file.write(f"window_amount={window_amount}\n")
        file.write(f"pulse_width_ns={pulse_width_ns}\n")
        file.write(f"height={height}\n")
    print(f"Variable values written to {config_file}")

# Writes fixed default values to the configuration file
def write_default_values_to_file():
    global window_size, max_peaks, window_amount, pulse_width_ns, height
    config_file = "default_values.txt"

    with open(config_file, "w") as file:
        file.write("window_size=12500\n")
        file.write("max_peaks=13\n")
        file.write("window_amount=100\n")
        file.write("pulse_width_ns=15\n")
        file.write("height=0.2\n")

    window_size = 12500
    max_peaks = int(13)
    window_amount = 100
    pulse_width_ns = 15
    height = 0.2

    print(f"Default values written to {config_file}")

# Reads the values from the text file
# If the file doesn't exist, creates it with default values
### Returns values array
def read_values_from_file():
    values = {}
    if os.path.exists(config_file):
        with open(config_file, "r") as file:
            for line in file:
                key, value = line.strip().split("=")
                values[key] = float(value) if '.' in value else int(value)
    else:
        print(f"{config_file} not found. Creating it with default values.")
        write_values_to_file()  # Create the file with default values
        values['window_size'] = 12500
        values['max_peaks'] = int(13)
        values['window_amount'] = 100
        values['pulse_width_ns'] = 15
        values['height'] = 0.2

    return values

# Shows the values read from the text file
def show_values_from_file(values):
    print(f"\nValues from the file:")
    print(f"{BLUE}\t- window_size: {values['window_size']}")
    print(f"\t- max_peaks: {values['max_peaks']}")
    print(f"\t- window_amount: {values['window_amount']}")
    print(f"\t- pulse_width_ns: {values['pulse_width_ns']}")
    print(f"\t- height: {values['height']}\n{RESET}")

# Writes WINDOW_SIZE straight into the file since the references don't work
def write_window_size_in_paths_script(window_size):
    config_file = "PathsAndConfigs.py"
    with open(config_file, "r") as f:
        lines = f.readlines()

    # Find and replace WINDOW_SIZE definition
    new_lines = []
    for line in lines:
        if line.startswith("WINDOW_SIZE = "):
            new_lines.append(f"WINDOW_SIZE = {window_size}\n")  # New value
        else:
            new_lines.append(line)

    # Write changes back to the file
    with open(config_file, "w") as f:
        f.writelines(new_lines)

# Deletes all .NPZ files from specific directories
def delete_npz_files():
    paths = [
        MANUAL_LABEL_READ_PATH,
        CONVERTED_WFM_PATH,
        TRAINING_DATA_INPUT_PATH,
        VAL_DATA_INPUT_PATH,
    ]

    for path in paths:
        npz_files = get_all_filepaths_from_directory(path, file_extension="npz")
        for file in npz_files:
            try:
                os.remove(file)
                print(f"Deleted: {file}")
            except Exception as e:
                print(f"Error deleting {file}: {e}")

    print("Deletion process completed.")

# Copies all files from CONVERTED_WFM_PATH to MANUAL_LABEL_READ_PATH,
# Then 80% to TRAINING_DATA_INPUT_PATH, and 20% to VAL_DATA_INPUT_PATH
def copy_converted_files():

    # Get all files in CONVERTED_WFM_PATH
    all_files = get_all_filepaths_from_directory(CONVERTED_WFM_PATH)

    if not all_files:
        print("No files found in CONVERTED_WFM_PATH. Skipping copy process.")
        return

    # Split files into 80% training and 20% validation
    total_files = len(all_files)
    split_index = int(total_files * 0.8)

    training_files = all_files[:split_index]  # First 80%
    val_files = all_files[split_index:]  # Last 20%

    # Copy all files to MANUAL_LABEL_READ_PATH
    # copy_all_files_from_dir_to_dir(CONVERTED_WFM_PATH, MANUAL_LABEL_READ_PATH)
    for file in all_files:
        shutil.copy(file, MANUAL_LABEL_READ_PATH)

    # Copy training files
    for file in training_files:
        shutil.copy(file, TRAINING_DATA_INPUT_PATH)

    # Copy validation files
    for file in val_files:
        shutil.copy(file, VAL_DATA_INPUT_PATH)

    print(f"Copied {total_files} files from CONVERTED_WFM_PATH.")
    print(f"→ {len(training_files)} files to TRAINING_DATA_INPUT_PATH.")
    print(f"→ {len(val_files)} files to VAL_DATA_INPUT_PATH.")
    print("Copy process completed.")


##################
# User Interface #
##################


# Show the values currently assigned to the variables
def show_values():
    print(f"\nCurrent values and their types:")
    print(f"{BLUE}\t- window_size value: {window_size}\t\ttype: {type(window_size)}")
    print(f"\t- max_peaks value: {max_peaks}\t\t\ttype: {max_peaks}")
    print(f"\t- window_amount value: {window_amount}\t\ttype: {type(window_amount)}")
    print(f"\t- pulse_width value: {pulse_width_ns}\t\t\ttype: {type(pulse_width_ns)}")
    print(f"\t- height value: {height}\t\t\t\ttype: {type(height)}\n{RESET}")

# Asks the user to confirm whether to use the saved values or the default ones, and also if they want to change them
def start_menu():
    global window_size, max_peaks, window_amount, pulse_width_ns, height  # Declare global variables here

    # Read current values from the file
    values_from_file = read_values_from_file()

    if values_from_file:
        show_values_from_file(values_from_file)
        read_and_apply_values_from_file()
    else:
        print("No saved values found, using default values.")

    print("What would you like to do?")

    print(f"\n{GREEN}1) Use SAVED values")
    print("2) Use DEFAULT values")
    print(f"3) CHANGE the values{RESET}")

    # Alter these values if the defaults were to change
    print(f"\nThe default values are:\n\t{BLUE}- window_size: 12500, "
          f"max_peaks: 13, window_amount: 100, "
          f"pulse_width_ns: 15, height: 0.2\n{RESET}")
    # print("\nThe default values are:")
    # print(f"\t- window_size: {window_size}")
    # print(f"\t- max_peaks: {max_peaks}")
    # print(f"\t- window_amount: {window_amount}")
    # print(f"\t- pulse_width_ns: {pulse_width_ns}")
    # print(f"\t- height: {height}\n")

    input_choice = input("Input: ")

    match input_choice:
        case "1":
            # Use the values from the file
            window_size = values_from_file['window_size']
            max_peaks = values_from_file['max_peaks']
            window_amount = values_from_file['window_amount']
            pulse_width_ns = values_from_file['pulse_width_ns']
            height = values_from_file['height']
            print("Using saved values.")
        case "2":
            # Use the default values and write them to the file
            write_default_values_to_file()
            write_window_size_in_paths_script(window_size)
            print("Using default values.")
        case "3":
            # Change the default values
            change_current_values()
        case _:
            print("Invalid input. Please try again.")
            start_menu()


    # Proceed to the next step after changing or confirming values
    main_menu()

# Run the pipeline script based on user choice
def main_menu():
    show_values()
    print("Which script would you like to run? (Press a number and then Enter)")
    print(f"{RED}\tNOTE: Options 1, 4, 5 end the process to avoid using cached data on 4, 5, 6, 7\n{RESET}")
    print("- DATA PREPARATION")
    print(f"{GREEN}\t1) ProcessRawWFM.py")
    print("\t2) ManualLabeling.py")
    print(f"\t3) AutomaticLabeling.py{RESET}")
    print("- MODEL TRAINING")
    print(f"{GREEN}\t4) TrainingDataGeneration.py")
    print(f"\t5) TrainingExample.py{RESET}")
    print("- DATA ANALYSIS")
    print(f"{GREEN}\t6) EvaluateDatasetWithModel.py")
    print(f"\t7) Stats.py{RESET}")
    print("- UTILITIES")
    print(f"{GREEN}\t8) CHANGE TRAINING VALUES")
    print("\t9) Show current parameter values")
    print("\t10) Split .NPZ files 80/20 for training")
    print(f"\t11) DELETE ALL .NPZ FILES{RESET}")
    print(f"\n(Choose 0 to exit the program during choices){RESET}")

    input_script_choice = input("Input: ")

    match input_script_choice:
        case "0":
            print("Exiting program...")
            exit()
        case "1":
            start_processing()
            # run_script_choices()
            exit()
        case "2":
            print(f"{RED}Notes: \n - This script may not work properly when running it through "
                f"PyCharm, try through Command Line if that's the case.{RESET}")
            print(f"{RED}- ManualLabeling.py relies of WINDOW_SIZE to show peaks,"
                  f"so if you are labeling size 100, be sure to set it to that.{RESET}")

            print("\nWould you like to proceed?")
            input_manual = input("Input (y/n): ")
            match input_manual:
                case "0":
                    print("Exiting program...")
                    exit()
                case "y":
                    subprocess.run(["python", "ManualLabeling.py"])
                case "n":
                    main_menu()
        case "3":
            start_automatic_labeling()
            main_menu()
            # exit()
        case "4":
            ml_data_generation()
            # run_script_choices()
            exit()
        case "5":
            start_training()
            main_menu()
            # exit()
        case "6":
            subprocess.run(["python", "EvaluateDatasetWithModel.py"])
            main_menu()
        case "7":
            subprocess.run(["python", "Stats.py"])
            main_menu()
        case "8":
            change_current_values()
            main_menu()
            # exit()
        case "9":
            show_values()
            main_menu()
        case "10":
            copy_converted_files()
            main_menu()
        case "11":
            delete_npz_files()
            main_menu()

        case _:
            print("Invalid input. Please try again: \n")
            main_menu()

# 1) Calls ProcessRawWFM.py with pulse_width_ns and height
def start_processing():
    try:
        process_wfm(pulse_width_ns, height)
    except Exception as e:
        print(f"Error occurred in ProcessRawWFM: {e}")
    # run_script_choices()

# 3) Calls AutomaticLabeling.py with pulse_width_ns and height
def start_automatic_labeling():
    try:
        autolabel(pulse_width_ns, height)
    except Exception as e:
        print(f"Error in AutomaticLabeling: {e}")
    # run_script_choices()

# 4) Calls TrainingDataGeneration.py with max_peaks as parameter
# (WINDOW_SIZE comes straight from PathsAndConfigs.py)
def ml_data_generation():
    print(f"{RED}\nDid you separate the NPZ files 80% in the folder training_data_input and 20% in the folder val_data_input?{RESET}")
    print("\t(Also, remember to delete old NPZ files.)")
    input_window_choice = input("Input (y/n): ")
    match input_window_choice:
        case "0":
            print("Exiting program...")
            exit()
        case "y":
            try:
                # run_datagen(max_peaks, int(window_size))
                run_datagen(max_peaks)

                print("Finished data generation")
            except Exception as e:
                print(f"Error in TrainingDataGeneration: {e}")
        case "n":
            main_menu()
        case _:
            print("Invalid input. Please try again: \n")
            ml_data_generation()

    # run_script_choices()

# 5) Calls TrainingExample.py
def start_training():
    try:
        run_training()
        print("Finished training")
    except Exception as e:
        print(f"Error in TrainingExample: {e}")
    exit()
    # run_script_choices()

# 8) Prompts the user to change variables, then writes them into default_values.txt
# This is also a submenu from the starting menu
def change_current_values():

    # Calls the functions to get new values and writes them to the file.
    show_values()
    print("Which parameters would you like to change? (Press a number and then Enter)\n")
    print(f"{GREEN}1) WINDOW_SIZE and max_peaks")
    print("2) pulse_width_ns AND height ")
    print("3) max_peaks (for testing)")
    print("\n4) Use default values\n")
    print("5) Save changes and proceed")
    print(f"6) Discard changes and proceed\n{RESET}")

    print("Use the following for reference to set the value of max_peaks:")
    print(f"AMOUNT_OF_CLASSES from the last dataset trained: {AMOUNT_OF_CLASSES}\n")

    input_parameter = input("Input: ")
    match input_parameter:
        case "0":
            print("Exiting program...")
            exit()
        case "1":
            ask_window_size()  # Get window size and max_peaks
            change_current_values()
        case "2":
            ask_width_height()  # Get window_amount, pulse_width_ns, height
            change_current_values()
        case "3":
            ask_max_peak()
            change_current_values()
        case "4":
            write_default_values_to_file()
            write_window_size_in_paths_script(window_size)
            change_current_values()
        case "5":
            write_values_to_file()  # Write the updated values to the file
            main_menu()
        case "6":
            main_menu()

    print("\nThe values have been updated.\n\n")
    print(f"\n\t{RED}!!! REMEMBER TO HAVE BACKUP DATASETS (training / eval) TO AVOID ERRORS BETWEEN TESTS !!!\n\n{RESET}")
    # run_script_choices()


#######################
# Variable Management #
#######################


###   Submenus from change_current_values()   ####

# Changes WINDOW_SIZE and max_peaks
def ask_window_size():
    global window_size, max_peaks, window_amount

    show_values()
    print("\nUse which window size?")
    print("- Suggestions: 16, 32, 64, 128, 256, 512, 1024, 2048, 4096")
    print("DEFAULT = 100\n")
    print("(Press 1 to go back \nPress 0 to exit the program)")

    # Applying user input

    temp_input = input("Input: ")
    match temp_input:
        case "0":
            print("Exiting program...")
            exit()
        case "1":
            main_menu()
        # case "16":
        #     window_size = int(78125)
        #     max_peaks = 78
        # case "32":
        #     window_size = int(39062)
        #     max_peaks = 40
        # case "64": # only trained with 18
        #     window_size = int(19531)
        #     max_peaks = 20
        # case "100": # standard
        #     window_size = int(12500)
        #     max_peaks = 13
        # case "128":
        #     window_size = int(9765)
        #     max_peaks = 12
        # case "256":
        #     window_size = int(4882)
        #     max_peaks = 8
        # case "512":
        #     window_size = int(2441)
        #     max_peaks = 5
        # case "1024":
        #     window_size = int(1220)
        #     max_peaks = 3
        # case "2048":
        #     window_size = int(610)
        #     max_peaks = 2
        # case "4096":
        #     window_size = int(305)
        #     max_peaks = 2
        case _:
            if temp_input.isdigit():
                # window_size = max(1, (12500 * 100) // int(temp_input))
                window_size = int(temp_input)
                ask_max_peak()
            else:
                ask_window_size()

    window_amount = temp_input
    # max_peaks = AMOUNT_OF_CLASSES
    # print(f"For {window_amount} windows, window size is {window_size} and max_peaks is {max_peaks}")
    TrainingDataGeneration.max_peaks = max_peaks
    write_window_size_in_paths_script(window_size)

    #run_script_choices()

# Changes pulse_width_ns and height
def ask_width_height():
    global pulse_width_ns, height
    show_values()
    print("Which parameters would you like to use? (Press a number and then Enter)")
    print("1) pulse_width_ns = 6 and height = 0.1")
    print("2) pulse_width_ns = 6 and height = 0.2")
    print("3) pulse_width_ns = 15 and height = 0.1")
    print("4) pulse_width_ns = 15 and height = 0.2 --- DEFAULT\n")
    print("5) Enter own values")
    print("6) Go Back")

    input_parameter = input("Input: ")

    match input_parameter:
        case "0":
            print("Exiting program...")
            exit()
        case "1":
            pulse_width_ns = 6
            height = 0.1
        case "2":
            pulse_width_ns = 6
            height = 0.2
        case "3":
            pulse_width_ns = 15
            height = 0.1
        case "4":
            pulse_width_ns = 15
            height = 0.2
        case "5":
            print("Enter pulse width in ns (int):")
            pulse_width_ns = int(input("Input: "))
            print("Enter height (float):")
            height = float(input("Input: "))
        case "6":
            main_menu()
        case _:
            print("Invalid input. Please try again: \n")
            ask_width_height()
    #run_script_choices()

# Changes max_peak
def ask_max_peak():
    global max_peaks

    print("\nThe variable max_peaks represents the number of classes in the model plus one:")
    print("e.g. if you enter 2, you get three classes (0, 1 and 2-peak examples)")
    print("\nEnter max_peak (int):")
    max_peaks = int(input("Input: "))

if __name__ == "__main__":
    start_menu()
    main_menu()