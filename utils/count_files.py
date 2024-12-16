import os


def count_JPG_files(directory):
    file_count = 0
    jpg_count = 0

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_count += 1
            if file.lower().endswith(".jpg"):
                jpg_count += 1

    return file_count, jpg_count


file_count, jpg_count = count_JPG_files(
    os.path.join(
        os.path.expanduser("~/dx"), "/home/sakamoto/dx/data/dataset0a/test"
    )
)
print("all files: ", file_count)
print("jpg files: ", jpg_count)
