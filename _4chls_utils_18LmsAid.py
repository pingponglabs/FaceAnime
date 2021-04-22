import numpy as np
import matplotlib.pyplot as plt
import torch, cv2
from io_utils import _load, _numpy_to_cuda, _numpy_to_tensor
from common.depth_loader import DepthImageLoader # zyt add

def map_2d_18pts_2d(lms2d_68):
    _18_indx_3d22d = [17, 19, 21, 22, 24, 26, 36, 40, 39, 42, 46, 45, 31, 30, 35, 48, 66, 54]
    lms2d = lms2d_68[:,_18_indx_3d22d]
    lms2d[:,7] = (lms2d_68[:,37] + lms2d_68[:,40])/2
    lms2d[:,10] = (lms2d_68[:,43] + lms2d_68[:,46])/2
    lms2d[:,16] = (lms2d_68[:,62] + lms2d_68[:,66])/2
    return lms2d

def CenterLabelHeatMap(img_width, img_height, c_x, c_y, sigma):
    X1 = np.linspace(1, img_width, img_width)
    Y1 = np.linspace(1, img_height, img_height)
    [X, Y] = np.meshgrid(X1, Y1)
    X = X - c_x
    Y = Y - c_y
    D2 = X * X + Y * Y
    E2 = 2.0 * sigma * sigma
    Exponent = D2 / E2
    heatmap = np.exp(-Exponent)
    return heatmap


def gen_gaussian_maps(joints, grid_x=120, grid_y=120, sigma=1.5):
    # print "Target generation -- Gaussian maps"
    num = joints.shape[0]
    heatmaps = np.zeros([num, grid_x, grid_y])
    for idx in range(num):
        heatmaps[idx] = CenterLabelHeatMap(grid_x, grid_y, joints[idx, 0], joints[idx, 1], sigma)
    heatmaps = sum(heatmaps)

    return heatmaps

def obtain_map(name, depth_img_loader, lst=True, img_size=64, lms_number = 'all'):
    if(lst):
        # import pdb; pdb.set_trace()
        # lms = lms.strip()
        # xx = lms.split(',')[1:]
        # name = lms.split(',')[0]
        depth_img, lms_pts = depth_img_loader.load_from_name(name, lm_number='all') # zyt add
        lms_pts = lms_pts.astype(np.int) if lms_number!='all' else False
        # pts = np.empty([2, 68])
        # pts[0,:] = [xx[i] for i in range(len(xx)) if (i) % 2 == 0]
        # pts[1,:] = [xx[i] for i in range(len(xx)) if (i) % 2 == 1]
    else:
        pass #TODO adding load from non-list object
        # lms_pts = lms.cpu().data.numpy()
        # depth_img, lms_pts = depth_img_loader.load_from_name(lms) # zyt add
    if lms_number!='all': # generate Gaussian map 
        heatmap = gen_gaussian_maps(lms_pts.transpose(), sigma=2)
        heatmap = heatmap / heatmap.max()
        heatmap = heatmap * depth_img
    else:
        heatmap = depth_img

    heatmap = np.float32(np.stack((heatmap, heatmap, heatmap), axis=-1))
    # ptsMap = np.zeros([120, 120])-1
    # indx = np.int32(np.floor(pts))
    # indx[indx>119] = 119
    # indx[indx<1] = 1
    # ptsMap[indx[1], indx[0]] = 1
    #
    # img = cv2.imread('./train_aug_120x120/' + lms.split(',')[0])
    # fig = plt.figure(figsize=plt.figaspect(.5))
    # ax = fig.add_subplot(1, 2, 1)
    # ax.imshow(img)
    # aa = heatmap
    # ax = fig.add_subplot(1, 2, 2)
    # ax.imshow(aa)

    # for ind in range(18):
    #     ax.plot(indx[0, ind], indx[1, ind], marker='o', linestyle='None', markersize=4, color='w',
    #             markeredgecolor='black', alpha=0.8)
    # ax.axis('off')
    #
    # plt.savefig(('./imgs/lms_18LmsMaps/' + lms.split(',')[0]))

    return cv2.resize(heatmap, (img_size, img_size), interpolation=cv2.INTER_CUBIC)

def obtain_map2(idx, pts, img):
    pts = map_2d_18pts_2d(pts)
    ptsMap = torch.zeros([120,120]) - 1
    indx = np.int32(pts.floor().int())
    indx[indx>119] = 119
    indx[indx<1] = 1
    ptsMap[indx[1], indx[0]] = 1
    '''
    fig = plt.figure(figsize=plt.figaspect(.5))
    ax = fig.add_subplot(1, 2, 1)
#    ax.imshow(img)
    aa = ptsMap
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(aa)

    for ind in range(18):
        ax.plot(indx[0, ind], indx[1, ind], marker='o', linestyle='None', markersize=4, color='w',
                markeredgecolor='black', alpha=0.8)
    ax.axis('off')

#    cv2.imwrite(('./imgs/bat3lmsMaps/' + 'xx.jpg'), ptsMap*255)
    plt.savefig(('./imgs/lms_18LmsMaps/' + '00_' + str(idx) + '.jpg'))
    '''
    return ptsMap

def comb_inputs(imgs, lmsMaps, permu=False):
    lmsMaps = np.array(lmsMaps).astype(np.float32)
    if permu == True:
        imgs = imgs.permute(0, 2, 3, 1)
    else:
        imgs = imgs
    outputs = [np.dstack((imgs[idx],lmsMaps[idx])) for idx in range(imgs.shape[0])]
    outputs = np.array(outputs).astype(np.float32)
    return outputs

def comb_inputs2(imgs, lmsMaps, permu=False):
    imgs = _numpy_to_cuda(imgs)
    if permu == True:
        imgs = imgs.permute(0, 2, 3, 1)
    else:
        imgs = imgs
    outputs = [torch.cat((imgs[idx], lmsMaps[idx].unsqueeze(2).cuda()), 2) for idx in range(imgs.shape[0])]
    return torch.stack(outputs)

def obtain_18pts_map2(pts, smp = True):
    _18_indx_2d22d = [0,1,2,3,4,5,6,7,8,9,10,11,13,14,15,17,18,19]
    if(smp == True):
        pts = pts[:, _18_indx_2d22d]
    else:
        pts = pts
#    ptsMap = np.zeros([120, 120])-1
#    indx = np.int32(np.floor(pts))
    ptsMap = torch.zeros([120,120]) - 1
    indx = np.int32(pts.floor().int())
    indx[indx>119] = 119
    indx[indx<1] = 1
    ptsMap[indx[1], indx[0]] = 1
    

    aa = ptsMap
    fig = plt.figure(figsize=plt.figaspect(.5))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(aa)
 
    for ind in range(18):
        ax.plot(indx[0, ind], indx[1, ind], marker='o', linestyle='None', markersize=4, color='w',
                markeredgecolor='black', alpha=0.8)
    ax.axis('off')

    cv2.imwrite(('./imgs/lms_18pts/' + 'xx.jpg'), ptsMap*255)

    return ptsMap
