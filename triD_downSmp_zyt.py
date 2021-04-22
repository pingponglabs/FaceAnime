import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os, cv2
import torch, pdb
import torch.nn as nn


### sampling mask
crop_face_mesh = sio.loadmat('./models/face_crop_region_mesh_3d_downSampling.mat')
reVerIndx = crop_face_mesh['verIndx2'] - 1
reVerIndx = reVerIndx.reshape(len(reVerIndx),)
reTri = crop_face_mesh['reTri2'] - 1
reTri = np.int32(reTri)

# ### dense mask
# crop_face_mesh = sio.loadmat('./models/face_crop_region_mesh2.mat')
# reVerIndx = crop_face_mesh['verIndx'] - 1
# reVerIndx = reVerIndx.reshape(len(reVerIndx),)
# reTri = crop_face_mesh['reTri'] - 1
# reTri = np.int32(reTri)

def _3D_downSmpl_test(src_img_path, src_ver_path, dst_ver_path):
    #pdb.set_trace()
    # img_name = src_path.split('_')[0]
    img = cv2.imread(src_img_path, cv2.COLOR_BGR2RGB)
    img = (img - 127.5)/128
    # import pdb; pdb.set_trace()
    vertices_src = sio.loadmat( src_ver_path)['ver']
    vertices_dst = sio.loadmat(dst_ver_path)['ver']

    colors = get_colors(img, vertices_src.transpose())
    colors = colors[reVerIndx,:]

    ver = vertices_dst[:,reVerIndx]
    ver = np.float64(ver)
    [h, w, c] = img.shape
    output = render_texture(ver, colors.transpose(), reTri, h, w, 3)
    return output, 0

def _3D_downSmpl(img_size, root_img, root_ver, src_path, dst_path):
    #pdb.set_trace()
    img_name = src_path.split(',')[0]
    img = cv2.imread(os.path.join(root_img, img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img - 127.5)/128
    vertices_src = sio.loadmat(os.path.join(root_ver, src_path.replace('.jpg', '.mat')))['vertex']
    vertices_dst = sio.loadmat(os.path.join(root_ver, dst_path.replace('.jpg', '.mat')))['vertex']

    colors = get_colors(img, vertices_src.transpose())
    colors = colors[reVerIndx,:]

    ver_src = vertices_src[:,reVerIndx]
    ver_src = np.float64(ver_src)

    ver_dst = vertices_dst[:,reVerIndx]
    ver_dst = np.float64(ver_dst)
    [h, w, c] = img.shape

    output_src = render_texture(ver_src, colors.transpose(), reTri, h, w, 3)
    output_dst = render_texture(ver_dst, colors.transpose(), reTri, h, w, 3)

    # output = cv2.resize(output, (img_size, img_size))

    return output_src, output_dst


def get_colors(image, vertices):
    '''
    Args:
        pos: the 3D position map. shape = (256, 256, 3).
    Returns:
        colors: the corresponding colors of vertices. shape = (num of points, 3). n is 45128 here.
    '''
    [h, w, _] = image.shape
    vertices[:,0] = np.minimum(np.maximum(vertices[:,0], 0), w - 1)  # x
    vertices[:,1] = np.minimum(np.maximum(vertices[:,1], 0), h - 1)  # y
    ind = np.round(vertices).astype(np.int32)
    colors = image[ind[:,1], ind[:,0], :] # n x 3
    return colors
    
    
def render_texture(vertices, colors, triangles, h, w, c = 3):
    ''' render mesh by z buffer
    Args:
        vertices: 3 x nver
        colors: 3 x nver
        triangles: 3 x ntri
        h: height
        w: width
    '''

    # initial
    image = np.zeros((h, w, c))

    depth_buffer = np.zeros([h, w]) - 999999.
    # triangle depth: approximate the depth to the average value of z in each vertex(v0, v1, v2), since the vertices are closed to each other
    tri_depth = (vertices[2, triangles[0,:]] + vertices[2,triangles[1,:]] + vertices[2, triangles[2,:]])/3.
    tri_tex = (colors[:, triangles[0,:]] + colors[:,triangles[1,:]] + colors[:, triangles[2,:]])/3.

    for i in range(triangles.shape[1]):
        tri = triangles[:, i] # 3 vertex indices

        # the inner bounding box
        umin = max(int(np.ceil(np.min(vertices[0,tri]))), 0)
        umax = min(int(np.floor(np.max(vertices[0,tri]))), w-1)

        vmin = max(int(np.ceil(np.min(vertices[1,tri]))), 0)
        vmax = min(int(np.floor(np.max(vertices[1,tri]))), h-1)

        if umax<umin or vmax<vmin:
            continue

        for u in range(umin, umax+1):
            for v in range(vmin, vmax+1):
                if tri_depth[i] > depth_buffer[v, u] and isPointInTri([u,v], vertices[:2, tri]):
                    depth_buffer[v, u] = tri_depth[i]
                    image[v, u, :] = tri_tex[:, i]
    return image


def isPointInTri(point, tri_points):
    ''' Judge whether the point is in the triangle
    Method:
        http://blackpawn.com/texts/pointinpoly/
    Args:
        point: [u, v] or [x, y]
        tri_points: three vertices(2d points) of a triangle. 2 coords x 3 vertices
    Returns:
        bool: true for in triangle
    '''
    tp = tri_points

    # vectors
    v0 = tp[:,2] - tp[:,0]
    v1 = tp[:,1] - tp[:,0]
    v2 = point - tp[:,0]

    # dot products
    dot00 = np.dot(v0.T, v0)
    dot01 = np.dot(v0.T, v1)
    dot02 = np.dot(v0.T, v2)
    dot11 = np.dot(v1.T, v1)
    dot12 = np.dot(v1.T, v2)

    # barycentric coordinates
    if dot00*dot11 - dot01*dot01 == 0:
        inverDeno = 0
    else:
        inverDeno = 1/(dot00*dot11 - dot01*dot01)

    u = (dot11*dot02 - dot01*dot12)*inverDeno
    v = (dot00*dot12 - dot01*dot02)*inverDeno

    # check if point in triangle
    return (u >= 0) & (v >= 0) & (u + v < 1)


# ver_path = './train.configs_v2/3dmm_params_grdth_from_2dasl/2dasl_vers/3dLandmarks_proj_resnet50/'
# img_path = './train_aug_120x120/'
#
#
# # from os import listdir
# # from os.path import isfile, join
# #
# # aa = listdir(ver_path)
# # idx = 0
# # for iitem in aa:
# #     idx = idx + 1
# #     print(idx)
# #     vertices = sio.loadmat(ver_path + iitem)['vertex']
# #     img = plt.imread(img_path + iitem.replace('.mat', '.jpg'))
# #     img = img / 255.0
# #     xxx = _3D_downSmpl(img, vertices)
# #
# #     plt.savefig('train.configs_v2/3dmm_params_grdth_from_2dasl/3d_downSmp_res/' + iitem.replace('.mat', '.jpg'))


def mask_fusion(inputs, _3dMask, x_dst, mms):
    ori_size = inputs.size()[-2:]
    inputs = nn.functional.interpolate(inputs, _3dMask[0].shape[:2])

    _3dMask = torch.tensor(np.array(_3dMask)).cuda()
    _3dMask = _3dMask.permute(0,3,1,2)
    aa = torch.sum(_3dMask, 1)
    bb = torch.stack([aa, aa, aa], 1)

    mm = torch.zeros(bb.shape).cuda()
    mm[bb!=0] = 1
    # mm = mms.float().cuda()
    _3dMask = _3dMask.float()
    fus_out = inputs.mul(1-mm) + _3dMask.mul(mm)
    fus_out = nn.functional.interpolate(fus_out, ori_size)

    # mms = torch.tensor(np.array(mms))
    # mms = mms.permute(0,3,1,2)
    # mm = mms.float().cuda()
    #
    # fus_out = inputs.mul(1-mm) + x_dst.mul(mm)
#     fus_out = inputs + _3dMask

    # sio.savemat('x_fake.mat', {'data': inputs.permute(0, 2, 3, 1).detach().cpu().numpy()})
    # sio.savemat('3dMask.mat', {'data': _3dMask.permute(0, 2, 3, 1).cpu().numpy()})
    # sio.savemat('dstImg.mat', {'data': x_dst.permute(0, 2, 3, 1).cpu().numpy()})
    # sio.savemat('mms.mat', {'data': mms.permute(0, 2, 3, 1).cpu().numpy()})


#    for idx in range(len(fus_out)):
#    #
#        plt.figure()
#        plt.imshow(_3dMask1[idx].permute(1,2,0).cpu().numpy())
#        plt.savefig('xxx.jpg')
    #     plt.subplot(1, 3, 1)
    #     inputs = inputs*0.5+0.5
    #     plt.imshow(inputs[idx].permute(1, 2, 0).cpu().numpy())
    #     plt.subplot(1, 3, 2)
    #     x_dst = x_dst*0.5+0.5
    #     plt.imshow(x_dst[idx].permute(1,2,0).cpu().numpy())
    #     plt.subplot(1, 3, 3)
    #     _3dMask = _3dMask*0.5+0.5
    #     plt.imshow(_3dMask[idx].permute(1,2,0).cpu().numpy())
    #
    #     plt.savefig(str(idx) + '.jpg')

    return fus_out
