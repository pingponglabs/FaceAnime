import numpy as np

# joints 5x2

def gen_gaussian_maps(joints, visibility, stride=8, grid_x=46, grid_y=46, sigma=7):
    #print "Target generation -- Gaussian maps"

    joint_num = joints.shape[0]
    gaussian_maps = np.zeros((joint_num + 1, grid_y, grid_x))
    for ji in range(0, joint_num):
        if visibility[ji]:
            gaussian_map = gen_single_gaussian_map(joints[ji, :], stride, grid_x, grid_y, sigma)
            gaussian_maps[ji, :, :] = gaussian_map[:, :]

    # Get background heatmap
    max_heatmap = gaussian_maps.max(0)
	
    gaussian_maps[joint_num, :, :] = 1 - max_heatmap

    return gaussian_maps
