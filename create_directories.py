import os

# Define the directory structure
directories = [
    "data/train/pool",
    "data/train/gym",
    "data/validation/pool",
    "data/validation/gym",
    "data/test/pool",
    "data/test/gym"
]

# Create the directories
for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"Created directory: {directory}")
