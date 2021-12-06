import os
import json


def get_objs(path: str) -> dict:
    item_dictionary = {}
    for item in os.walk(path):
        for obj in item[2]:
            if obj.endswith(".obj") and obj[0].isdigit():
                item_dictionary[os.path.join(item[0], obj)] = int(obj.split("-")[0])
    return item_dictionary


if __name__ == "__main__":
    items = get_objs("/home/avena/Dropbox/synth_dataset/BlenderProc/avena/obj/Containers")

    with open("items/containers_json.json", "w") as f:
        json.dump(items, f, indent=2)
