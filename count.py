import os
import argparse


def count_lines_in_txt_files(directory_path):
    """
    Counts the number of lines in each .txt file within a given directory.

    Args:
        directory_path (str): The path to the directory containing the .txt files.
    """
    try:
        if not os.path.exists(directory_path):
            print(f"Error: Directory '{directory_path}' does not exist.")
            return

        total_lines = 0
        file_count = 0

        for filename in os.listdir(directory_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(directory_path, filename)
                try:
                    with open(file_path, "r") as file:
                        lines = file.readlines()
                        num_lines = len(lines)
                        print(f"File: {filename}, Lines: {num_lines}")
                        total_lines += num_lines
                        file_count += 1
                except FileNotFoundError:
                    print(f"Error: File '{file_path}' not found.")
                except Exception as e:
                    print(f"Error reading file '{file_path}': {e}")
        if file_count > 0:
            print(f"Total lines in {file_count} .txt files: {total_lines}")
        else:
            print(f"No .txt files found in directory: {directory_path}")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Count lines in .txt files within a directory."
    )
    parser.add_argument(
        "directory", type=str, help="The directory path containing .txt files."
    )
    args = parser.parse_args()

    directory_path = args.directory
    count_lines_in_txt_files(directory_path)
