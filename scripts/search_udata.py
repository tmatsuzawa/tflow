## Author: ChatGPT3.5
## Example use:
## python search_files.py keyword1 keyword2 /path/to/directory/
## python search_udata.py /Volumes/mercury/udata/piv_data front propagation  --avoid_words ensemble
import os
import argparse

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="Search for files in a directory and its subdirectories based on keyword presence in filenames")

# Add arguments for the directory, keywords, output directory, and avoid words
parser.add_argument("directory", type=str, help="Directory to search for files")
parser.add_argument("keywords", nargs="+", type=str, help="Keywords to search for in filenames")
parser.add_argument("--output_dir", type=str, default="/Volumes/mercury/projects", help="Output directory to save results (default: /Volumes/mercury/projects)")
parser.add_argument("--avoid_words", type=str, nargs="*", default=[], help="Avoid filenames containing these words")

# Parse the command-line arguments
args = parser.parse_args()

# Get the directory, keywords, output directory, and avoid words from the parsed arguments
directory = args.directory
keywords = args.keywords
output_dir = args.output_dir
avoid_words = args.avoid_words

# Generate the output file name based on the keywords
output_file = os.path.join(output_dir, f"output_{'_'.join(keywords)}.txt")

# Create a list to store the filenames
filenames = []

# Loop through all files and directories in the directory and its subdirectories recursively
for root, _, files in os.walk(directory):
    for filename in files:
        file_path = os.path.join(root, filename)
        # Check if all keywords are present in the file path and none of the avoid words are present
        if all(keyword in file_path for keyword in keywords) and any(avoid_word not in file_path for avoid_word in avoid_words):
            filenames.append(os.path.abspath(file_path))

# Sort the filenames alphabetically
filenames.sort()

# Open the output file in write mode
with open(output_file, 'w') as file:
    # Write the sorted filenames to the output file
    for filename in filenames:
        file.write(f"{filename}\n")

# Print a message indicating the search is complete
print("Search complete. Results written to", output_file)

