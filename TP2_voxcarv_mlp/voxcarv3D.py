
import matplotlib.image as mpimg
import numpy as np
from skimage import measure
import trimesh

# Camera Calibration for Al's image[0..11].pgm
calib=np.array([[-230.924, 0, -33.6163, 300,  -78.8596, -178.763, -127.597, 300,  -0.525731, 0, -0.85065, 2],
[-178.763, -127.597, -78.8596, 300,  0, -221.578, 73.2053, 300,  0, -0.85065, -0.525731, 2],
[-73.2053, 0, -221.578, 300,  78.8596, -178.763, -127.597, 300,  0.525731, 0, -0.85065, 2],
[-178.763, 127.597, -78.8596, 300,  0, 33.6163, -230.924, 300,  0, 0.85065, -0.525731, 2],
[73.2053, 0, 221.578, 300,  -78.8596, -178.763, 127.597, 300,  -0.525731, 0, 0.85065, 2],
[230.924, 0, 33.6163, 300,  78.8596, -178.763, 127.597, 300,  0.525731, 0, 0.85065, 2],
[178.763, -127.597, 78.8596, 300,  0, -221.578, -73.2053, 300,  0, -0.85065, 0.525731, 2],
[178.763, 127.597, 78.8596, 300,  0, 33.6163, 230.924, 300,  0, 0.85065, 0.525731, 2],
[-127.597, -78.8596, 178.763, 300,  -33.6163, -230.924, 0, 300,  -0.85065, -0.525731, 0, 2],
[-127.597, 78.8596, 178.763, 300,  -221.578, -73.2053, 0, 300,  -0.85065, 0.525731, 0, 2],
[127.597, 78.8596, -178.763, 300,  221.578, -73.2053, 0, 300,  0.85065, 0.525731, 0, 2],
[127.597, -78.8596, -178.763, 300,  33.6163, -230.924, 0, 300,  0.85065, -0.525731, 0, 2]])

####### MAIN #########  
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Simple demo to of the vox carving algorithm.')
    parser.add_argument('-r', '--grid_resolution', type=int, default=100,
                        help='discretization of the volume')
    args = parser.parse_args()

    resolution=args.grid_resolution

    ################################
    # Initialization of structures #

    # Build 3D grids
    # 3D Grids are of size: resolution x resolution x resolution/2
    step = 2/resolution

    # The grids X, Y, Z allow to query the world position inside the volumetric representation:
    # given a volume location i,j,k, then X[i,j,k] gives it's x-world-coordinate
    X, Y, Z = np.mgrid[-1:1:step, -1:1:step, -0.5:0.5:step]

    # This volume defines the occupancy
    occupancy = np.ndarray(shape=(resolution,resolution, int(resolution/2)), dtype=int)

    # Voxels are initially occupied then carved with silhouette information
    occupancy.fill(1)

    # the next lines set to empty the borders of the volume
    # so that marching cubes can compute a first full cube
    occupancy[0,:,:] = 0
    occupancy[-1,:,:] = 0
    occupancy[:,0,:] = 0
    occupancy[:,-1,:] = 0
    occupancy[:,:,0] = 0
    occupancy[:,:,-1] = 0

    #       End of initialization     #
    ###################################

    # TODO: create an array with all loaded images
    # Example of image loading for i = 1
    i=1
    myFile = "images/image{0}.pgm".format(i) # read the input silhouettes
    img = mpimg.imread(myFile)
    if img.dtype == np.float32: # if not integer
        img = (img * 255).astype(np.uint8)

    # Compute grid projection in images
    # TODO: TO BE COMPLETED
    # Loop over all grid elements in 3D

        # Loop over the cameras

            # Make sure you get the right camera calibration
            # and the associated image

            # Project point

            # Check if projection is in the silhouette

            # if not in the silhouette, set the occupancy to zero

    # Voxel visualization
    verts, faces, normals, values = measure.marching_cubes(occupancy, 0.25) # Marching cubes 
    surf_mesh = trimesh.Trimesh(verts, faces, validate=True) # Export in a standard file format 
    surf_mesh.export('output/alvoxels.off')

    print("Run \"python show_mesh.py output/alvoxels.off\" to see the results (or use MeshLab)")

    np.save("output/occupancy.npy", occupancy)