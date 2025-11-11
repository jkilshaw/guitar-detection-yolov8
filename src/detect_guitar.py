"""
YOLOv8 Guitar Detection Script
------------------------------
Processes all images in ~/Desktop/Guitar_Input/ using a trained YOLOv8 model.
Draws bounding boxes on detected guitars and saves results to ~/Desktop/Guitar_Output/.

Dependencies:
    pip install ultralytics opencv-python

Usage:
    python detect_guitar.py
    Processes all .jpg/.jpeg images from input folder
"""

import cv2
from pathlib import Path
from ultralytics import YOLO


def detect_guitar(image_path, model_path="runs/guitar_kaggle_finetune3/weights/best.pt",
                  output_path="output.jpg", conf_threshold=0.25):
    """
    Detect guitars in an image using a trained YOLOv8 model.

    Args:
        image_path (str): Path to the input image
        model_path (str): Path to the trained YOLOv8 model weights
        output_path (str): Path to save the output image
        conf_threshold (float): Confidence threshold for detections (0-1)

    Returns:
        int: Number of guitars detected (-1 if error)
    """


    # Validate input image exists
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"Error: Image file not found at '{image_path}'")
        return -1


    # Load the current trained model
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"Error: Model file not found at '{model_path}'")
        return -1

    print(f"Loading model from: {model_path}")
    model = YOLO(str(model_path))


    # Load the image through opencv
    print(f"Loading image: {image_path}")
    img = cv2.imread(str(image_path))

    if img is None:
        print(f"Error: Failed to load image from '{image_path}'")
        return -1

    img_height, img_width = img.shape[:2]
    print(f"Image size: {img_width}x{img_height}")

    # Run inference
    print(f"Running detection (confidence threshold: {conf_threshold})...")
    results = model.predict(
        source=str(image_path),
        conf=conf_threshold,
        verbose=False  # Suppress verbose output
    )


    # Process the detections
    # Get the first (and only) result
    result = results[0]

    # Extract bounding boxes, confidences, and class IDs
    boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes in [x1, y1, x2, y2] format
    confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
    class_ids = result.boxes.cls.cpu().numpy()  # Class IDs (should all be 0 for 'guitar')

    num_detections = len(boxes)


    # Draw the bounding boxes
    if num_detections > 0:
        print(f"Detected {num_detections} guitar(s)!")

        for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
            # Extract coordinates
            x1, y1, x2, y2 = map(int, box)

            # Get class name (should be 'guitar')
            class_name = model.names[int(cls_id)]

            # Create label with confidence
            label = f"{class_name} {conf:.2f}"

            print(f"  [{i+1}] {label} at [{x1}, {y1}, {x2}, {y2}]")

            # Draw bounding box (green color)
            cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

            # Draw label background
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            y1_label = max(y1, label_size[1] + 10)
            cv2.rectangle(
                img,
                (x1, y1_label - label_size[1] - 10),
                (x1 + label_size[0], y1_label + baseline - 10),
                color=(0, 255, 0),
                thickness=-1  # Filled rectangle
            )

            # Draw label text (black color on green background)
            cv2.putText(
                img,
                label,
                (x1, y1_label - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color=(0, 0, 0),
                thickness=2
            )


        # Save the output image
        cv2.imwrite(output_path, img)
        print(f"Saved annotated image to: {output_path}")

    else:
        print("No guitar detected")
        print(f"Saving unmodified image to: {output_path}")
        cv2.imwrite(output_path, img)

    return num_detections


def main():
    """
    Main function - processes all images from input folder.
    """


    # Configuration
    INPUT_FOLDER = Path.home() / "Desktop" / "Guitar_Input"
    OUTPUT_FOLDER = Path.home() / "Desktop" / "Guitar_Output"
    MODEL_PATH = "runs/guitar_kaggle_finetune3/weights/best.pt"
    CONFIDENCE_THRESHOLD = 0.25  # Adjust between 0.0 - 1.0


    # Create the output folder
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


    # Find all images in the input folder
    image_extensions = ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG']
    image_files = []

    for ext in image_extensions:
        image_files.extend(INPUT_FOLDER.glob(ext))

    # Sort for consistent processing order
    image_files = sorted(image_files)

    if not image_files:
        print("=" * 60)
        print("No images found in ~/Desktop/Guitar_Input")
        print("=" * 60)
        return


    # Loop through and process each image
    print("=" * 60)
    print("YOLOV8 GUITAR DETECTION - BATCH PROCESSING")
    print("=" * 60)
    print(f"Input folder: {INPUT_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"Found {len(image_files)} image(s) to process")
    print("=" * 60)

    total_guitars = 0
    successful_images = 0

    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {image_path.name}")
        print("-" * 60)

        # Create output filename with _detected suffix
        output_filename = image_path.stem + "_detected" + image_path.suffix
        output_path = OUTPUT_FOLDER / output_filename

        # Run detection
        num_guitars = detect_guitar(
            image_path=image_path,
            model_path=MODEL_PATH,
            output_path=str(output_path),
            conf_threshold=CONFIDENCE_THRESHOLD
        )

        if num_guitars >= 0:
            successful_images += 1
            total_guitars += num_guitars

    # Printed summary of results
    print("\n" + "=" * 60)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total images processed: {successful_images}/{len(image_files)}")
    print(f"Total guitars detected: {total_guitars}")
    print(f"Results saved to: {OUTPUT_FOLDER}")
    print("=" * 60)


if __name__ == "__main__":
    main()
