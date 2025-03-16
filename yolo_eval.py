import os
from ultralytics import YOLO

# Load the trained YOLO model (you can replace the path with your model's path)
model_path = "colab/best.pt"  # Update with your trained model's path
model = YOLO(model_path)

# Set the base directory where the data is located (assuming this is the base path)
base_path = "train"  # E.g., 'dataset_folder', replace with the correct base directory

test_images_path = os.path.join(base_path, "test/images/1")

# Perform inference on the test dataset
results = model.predict(test_images_path)

# Print the results
print("Inference Results:")
results.show()  # This will display the images with predicted bounding boxes

# Optionally, you can also print a summary of the results
results.print()  # This will print the model's performance metrics (e.g., mAP, precision, recall, etc.)

# Optionally, you can get more detailed information like predictions for each image
for result in results:
    print(f"Predictions for {result.path}:")
    print(
        result.pandas().xywh
    )  # This will print the predictions in a pandas DataFrame format
