import blenderproc
import json
import math
import numpy as np
import argparse

global table

parser = argparse.ArgumentParser()
parser.add_argument('output', nargs='?')
args = parser.parse_args()

def get_all_items_list(path_json: str) -> dict:
    with open(path_json, 'r') as f:
        return json.load(f)


def choose_items_to_load(items: dict, n: int) -> (list, list):
    ids = list(set(items.values()))
    ids_list = list(np.random.choice(ids, n))
    ids_list = list(map(int, ids_list))

    items_list = []
    for chosen_id in ids_list:
        matching_items = [k for k, v in items.items() if v == chosen_id]
        items_list.append(np.random.choice(matching_items, 1)[0])

    return items_list, ids_list


def sample_pose(obj_sampled: blenderproc.types.MeshObject) -> None:
    global table
    obj_sampled.set_location(blenderproc.sampler.upper_region(
        objects_to_sample_on=table,
        min_height=0.1,
        max_height=0.2,
        use_ray_trace_check=False
    ))
    obj_sampled.set_rotation_euler(np.random.uniform([0, 0, 0], [0.2, 0.2, np.pi * 2]))


def sample_pose_wrapper(obj_parent: blenderproc.types.MeshObject):
    def sample_pose_inside(obj_sampled_inside):
        obj_sampled_inside.set_location(blenderproc.sampler.upper_region(
            objects_to_sample_on=obj_parent,
            min_height=0.1,
            # max_height=0.4,
            use_ray_trace_check=True,
            upper_dir=[0.0, 0.0, 1.0],
        ))
        obj_sampled_inside.set_rotation_euler(np.random.uniform([0, 0, 0], [0.6, 0.6, np.pi * 2]))

    return sample_pose_inside


def main():
    global table
    # Load items from file
    container_items = get_all_items_list("/home/avena/PycharmProjects/pythonProject/items/containers_json.json")
    consumable_items = get_all_items_list("/home/avena/PycharmProjects/pythonProject/items/consumables_json.json")

    for _ in range(10000):
        n_containers = 10
        n_consumables = 60
        containers_to_load, ids_of_loaded_containers = choose_items_to_load(container_items, n_containers)
        consumables_to_load, ids_of_loaded_consumables = choose_items_to_load(consumable_items, n_consumables)

        blenderproc.init()
        blenderproc.utility.reset_keyframes()

        table = blenderproc.loader.load_blend("/home/avena/Dropbox/synth_dataset/BlenderProc/avena/Bez_fspy.blend")[0]
        sampler = sample_pose_wrapper(table)
        # Make table static
        table.enable_rigidbody(False)

        loaded_containers = [item for item_to_load in containers_to_load for item in blenderproc.loader.load_obj(item_to_load)]

        for item, item_id, name in zip(loaded_containers, ids_of_loaded_containers, containers_to_load):
            item.set_cp("category_id", item_id)
            item_name = name.split("/")[-1].split("-")[-1].split(".")[0]
            item.set_name(item_name)
            item.enable_rigidbody(True)

        blenderproc.object.sample_poses_on_surface(loaded_containers, table, sampler, min_distance=0.1, max_distance=10)

        # Simulate physics
        blenderproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=0.5, max_simulation_time=2,
                                                                check_object_interval=0.5, substeps_per_frame=3)

        light_p = blenderproc.types.Light()
        light_p.set_type("POINT")
        r = np.random.randint(2, 10)
        alpha = np.random.randint(0, 360)
        y = r * math.sin(math.pi / 180 * alpha)
        x = r * math.cos(math.pi / 180 * alpha)
        z = np.random.randint(2, 5)
        light_p.set_location([x, y, z])
        light_p.set_energy(1000)

        blenderproc.camera.set_intrinsics_from_blender_params(lens=0.017453, image_width=2400, image_height=1350,
                                                              lens_unit='FOV')

        position = [0, 0, 138]
        rotation = [0, 0, 0]
        matrix_world = blenderproc.math.build_transformation_mat(position, rotation)
        blenderproc.camera.add_camera_pose(matrix_world)

        loaded_consumables = [item for item_to_load in consumables_to_load for item in
                              blenderproc.loader.load_obj(item_to_load)]

        for item, item_id, name in zip(loaded_consumables, ids_of_loaded_consumables, consumables_to_load):
            item.set_cp("category_id", item_id)
            item_name = name.split("/")[-1].split("-")[-1].split(".")[0]
            item.set_name(item_name)
            item.enable_rigidbody(True)

        for loaded_container in loaded_containers:
            if "Bowl" in loaded_container.get_name() or "Plate" in loaded_container.get_name():
                loaded_container.enable_rigidbody(False, collision_shape='MESH')
            else:
                loaded_container.enable_rigidbody(False)

        containers_dict = {}
        for i in range(n_containers):
            containers_dict[i] = []
        for consumable in loaded_consumables:
            consumable.enable_rigidbody(True)
            on_which_container = np.random.randint(0, n_containers)
            containers_dict[on_which_container].append(consumable)

        for key, val in containers_dict.items():
            sampler = sample_pose_wrapper(loaded_containers[key])
            blenderproc.object.sample_poses_on_surface(val, loaded_containers[key], sampler,
                                                       min_distance=0.01, max_distance=10, max_tries=300,
                                                       )

        blenderproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=0.5, max_simulation_time=1.5,
                                                                check_object_interval=0.5, substeps_per_frame=3)

        # blenderproc.renderer.enable_normals_output()
        blenderproc.renderer.set_samples(30)
        data = blenderproc.renderer.render()
        seg_data = blenderproc.renderer.render_segmap(map_by=["instance", "class", "name"])
        blenderproc.writer.write_coco_annotations(args.output,
                                                  instance_segmaps=seg_data["instance_segmaps"],
                                                  instance_attribute_maps=seg_data["instance_attribute_maps"],
                                                  colors=data["colors"],
                                                  color_file_format="JPEG",
                                                  append_to_existing_output=True)


if __name__ == "__main__":
    main()
