"""
check_input_images.py

Checks the folder ~/Desktop/Guitar_Input and reports any files
that are NOT JPEGs (.jpg or .jpeg).
Can be run standalone or imported by other scripts.

"""

from pathlib import Path

# Return the path to ~/Desktop/Guitar_Input
def get_input_folder():
    folder = Path.home() / "Desktop" / "Guitar_Input"
    return folder

# Check if a file is truly JPEG and starts with the JPEG numbers
def is_real_jpeg(file_path: Path) -> bool:
    try:
        with open(file_path, "rb") as f:
            header = f.read(3)
        return header == b'\xFF\xD8\xFF'  # JPEG signature
    except Exception as e:
        print(f"Error reading {file_path.name}: {e}")
        return False

# Return a list of files that are not true JPEGs
def find_non_jpegs(folder_path):
    bad_files = []
    for file in folder_path.iterdir():
        if not file.is_file():
            continue
        if file.name.startswith('.'):  # Skip hidden files
            continue
        if not is_real_jpeg(file):
            bad_files.append(file)
    return bad_files

# Run the functions on the folder
def check_all_jpegs():
    folder = get_input_folder()

    if not folder.exists():
        print(f" Folder does not exist: {folder}")
        return False

    bad_files = find_non_jpegs(folder)

    if not bad_files:
        print(f" All files in {folder} are *real* JPEGs.")
        return True
    else:
        print(f" Found {len(bad_files)} non-JPEG or corrupted files:")
        for f in bad_files:
            print(f"   - {f.name}")
        print("➡ You should run your converter script on these.")
        return False

if __name__ == "__main__":
    check_all_jpegs()