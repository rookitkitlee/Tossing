import json
import csv

def fu_save_json(data, path):
    with open(path, 'w') as file_obj:
        json.dump(data, file_obj)

def fu_load_json(path):
    with open(path, 'r', encoding='utf8') as fp:
        json_data = json.load(fp)
        return json_data


def fu_load_csv(path):
    with open(path, 'r', encoding='utf8') as fp:
        reader = csv.reader(fp)
        data = [d for d in reader]
        return data

def fu_save_csv(head, data, path):
    with open(path, 'w', newline='') as t:
        writer = csv.writer(t)
        writer.writerow(head)
        writer.writerows(data)