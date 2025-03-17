import os


def count_lines_in_txt_files(directory):
    total_lines = 0

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                line_count = sum(1 for _ in file)
                total_lines += line_count
                print(f"{filename}: {line_count} lines")

    print(f"Total lines in all .txt files: {total_lines}")


# Set the directory path (change this as needed)
directory_path = "datasets/train/labels"  # Current directory
count_lines_in_txt_files(directory_path)
