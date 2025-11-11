"""
convert_to_jpeg.py
------------------
Converts non-JPEG images in ~/Desktop/Guitar_Input into true JPEGs.
Depends on functions from check_input_images.py

This script:
1. Finds all non-JPEG files using check_input_images functions
2. Converts each one to a proper JPEG using PIL/Pillow
3. Saves the new JPEG with .jpg extension
4. Optionally cleans up the original files
"""

from pathlib import Path
from PIL import Image
from check_input_images import get_input_folder, find_non_jpegs, is_real_jpeg

# Configuration: Set to True to delete originals after successful conversion
DELETE_ORIGINALS_AFTER_CONVERSION = True

def convert_image_to_jpeg(file_path: Path) -> Path:
    """
    Convert a single image file to JPEG format.

    Args:
        file_path: Path object pointing to the image to convert

    Returns:
        Path object to the newly created JPEG file

    Raises:
        Exception if conversion fails
    """
    # Open the image
    img = Image.open(file_path)
    # Ensure it's in RGB mode (JPEG needs RGB)
    if img.mode not in ("RGB",):
        # simplest/safer way for our case:
        img = img.convert("RGB")
    # Create new filename with .jpg extension
    new_path = file_path.with_suffix(".jpg")
    # Save as JPEG
    img.save(new_path, "JPEG", quality=95)
    return new_path

def handle_original_file(original_path: Path, delete: bool = False):
    """
    Handle the original non-JPEG file after successful conversion.

    Args:
        original_path: Path to the original file
        delete: If True, delete the file. If False, just report it.
    """
    if delete:
        try:
            original_path.unlink()
            print(f"  Deleted original: {original_path.name}")
        except Exception as e:
            print(f"  Could not delete {original_path.name}: {e}")
    else:
        # Let the user know it still exists
        print(f"  (kept original: {original_path.name})")

def main():
    print("= Starting JPEG conversion process...\n")

    # Get the input folder (~/Desktop/Guitar_Input)
    folder = get_input_folder()

    # Make sure it exists
    if not folder.exists():
        print(f" Input folder does not exist: {folder}")
        print("Create it and put images in there first.")
        return

    # Find all non-JPEG files using the checker script's logic
    non_jpegs = find_non_jpegs(folder)

    # If nothing to do, exit
    if not non_jpegs:
        print(f" No non-JPEG files found in {folder}. Nothing to convert.")
        return

    print(f"Found {len(non_jpegs)} file(s) to convert:\n")
    for f in non_jpegs:
        print(f"  - {f.name}")
    print()

    converted_count = 0
    failed_files = []

    # Loop through each bad file and try to convert
    for file_path in non_jpegs:
        try:
            new_path = convert_image_to_jpeg(file_path)
            print(f" Converted: {file_path.name} → {new_path.name}")
            converted_count += 1

            # handle original depending on config
            handle_original_file(file_path, delete=DELETE_ORIGINALS_AFTER_CONVERSION)

        except Exception as e:
            print(f" Failed to convert {file_path.name}: {e}")
            failed_files.append(file_path)

    # Give a summary of the results
    print("\n================ SUMMARY ================")
    print(f" Successfully converted: {converted_count} file(s)")
    if failed_files:
        print(f" Failed to convert: {len(failed_files)} file(s)")
        for f in failed_files:
            print(f"   - {f.name}")
    else:
        print(" No conversion failures.")
    print(f"📁 All JPEGs saved to: {folder}")
    if DELETE_ORIGINALS_AFTER_CONVERSION:
        print(" Originals were deleted after conversion.")
    else:
        print("ℹ Originals were kept. Set DELETE_ORIGINALS_AFTER_CONVERSION = True to remove them.")


if __name__ == "__main__":
    main()