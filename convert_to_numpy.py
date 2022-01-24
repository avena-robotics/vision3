import numpy as np
import open3d as o3d

pcd_load = o3d.io.read_point_cloud("1642598887745250449_scene.ply")

# Convert Open3D.o3d.geometry.PointCloud to numpy array
xyz_load = np.asarray(pcd_load.points, dtype=np.float32)
xyz_color = np.asarray(pcd_load.colors)

print(xyz_load.shape)
with open("pcld_orange.npy", 'wb') as f:
    np.save(f, {"xyz":xyz_load, "xyz_color":xyz_color})