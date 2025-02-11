# from ultralytics.data.converter import convert_coco
# convert_coco(labels_dir='./annotations/')

import os
from tqdm import tqdm

def remap_labels(input_folder, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Classes to keep (0, 2, 7, 5), and remap others to 1
    classes_to_keep = {0, 2, 7, 5}
    remapped_class = 1

    # List all txt files in the input folder
    label_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]

    for label_file in tqdm(label_files):
        input_path = os.path.join(input_folder, label_file)
        output_path = os.path.join(output_folder, label_file)

        with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
            for line in f_in:
                # Split line into parts
                parts = line.strip().split()
                if len(parts) != 5:
                    # Skip lines that do not have the expected format
                    continue

                class_id = int(parts[0])
                x, y, w, h = map(float, parts[1:])

                # Check if the class should be remapped
                if class_id in classes_to_keep:
                    if class_id in [2, 7, 5]:
                        class_id = remapped_class
                    # Write the modified line to the output file
                    f_out.write(f"{class_id} {x} {y} {w} {h}\n")

if __name__ == "__main__":
    input_folder = "./coco_converted/labels/train2017/"
    output_folder = "./coco_converted/remapped_labels/train2017/"
    
    remap_labels(input_folder, output_folder)
    print("Label files remapped successfully.")
