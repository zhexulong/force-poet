# Output CSV file
import csv
import json

input_file = 'output_book/results.json'
with open(input_file, 'r') as file:
    data = json.load(file)
output_file = 'output_book/results.csv'

# Function to convert nested list to space-separated string
def list_to_str(lst):
    return ' '.join(map(str, lst))

# Write to CSV
with open(output_file, 'w', newline='') as csvfile:
    fieldnames = ['scene_id', 'im_id', 'obj_id', 'score', 'R', 't', 'time', 'box']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for key, value in data.items():
        for obj_id, obj_data in value.items():
            scene_id = 48  # Assuming fixed scene_id
            im_id = 1     # Assuming fixed im_id
            score = 1.0   # Assuming fixed score
            R = list_to_str([elem for sublist in obj_data['rot'] for elem in sublist])
            t = list_to_str(obj_data['t'])
            box = list_to_str(obj_data['box'])
            time = 1.2229998111724854  # Assuming fixed time

            writer.writerow({
                'scene_id': scene_id,
                'im_id': im_id,
                'obj_id': obj_id,
                'score': score,
                'R': R,
                't': t,
                'time': time,
                'box': box
            })

print(f"Data has been written to {output_file}")