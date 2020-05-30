import numpy as np

import random


def run_ransac(data, estimate, is_inlier, sample_size, goal_inliers, max_iterations, stop_at_goal=True,
               random_seed=None):
    best_ic = 0
    best_model = None
    random.seed(random_seed)
    # random.sample cannot deal with "data" being a numpy array
    data = list(data)
    for i in range(max_iterations):
        s = random.sample(data, int(sample_size))
        m = estimate(s)
        ic = 0
        for j in range(len(data)):
            if is_inlier(m, data[j]):
                ic += 1

        print(s)
        print('estimate:', m, )
        print('# inliers:', ic)

        if ic > best_ic:
            best_ic = ic
            best_model = m
            if ic > goal_inliers and stop_at_goal:
                break
    print('took iterations:', i + 1, 'best model:', best_model, 'explains:', best_ic)
    return best_model, best_ic


def augment(xyzs):
    axyz = np.ones((len(xyzs), 4))
    axyz[:, :3] = xyzs
    return axyz


def estimate(xyzs):
    axyz = augment(xyzs[:3])
    return np.linalg.svd(axyz)[-1][-1, :]


def is_inlier(coeffs, xyz, threshold):
    return np.abs(coeffs.dot(augment([xyz]).T)) < threshold


if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    import cv2

    ox = 1280 / 2
    oy = 720 / 2
    fx = 1280 / (2 * np.tan(0.5 * np.pi * 90 / 180))
    fy = 720 / (2 * np.tan(0.5 * np.pi * 59 / 180))

    depth_bg = cv2.imread('data/depth_000149.png', -1)
    depth_bg[2600 < depth_bg] = 0
    gridyy, gridxx = np.mgrid[:720, :1280]

    xx_cam = (gridxx - ox) / fx * depth_bg
    yy_cam = (gridyy - oy) / fy * depth_bg

    depth_bg = depth_bg[..., np.newaxis]
    bg_xyz = np.concatenate((xx_cam[..., np.newaxis], yy_cam[..., np.newaxis], depth_bg), axis=2)
    bg_xyz_resh_all = np.reshape(bg_xyz, (-1, 3))
    bg_xyz_resh = bg_xyz_resh_all[bg_xyz_resh_all[:, 2] > 0]

    fig = plt.figure()
    ax = mplot3d.Axes3D(fig)


    def plot_plane(a, b, c, d):
        xx, yy = np.mgrid[:1280:100, :720:50]
        # xx, yy = np.mgrid[:100, :100]
        return xx, yy, (-d - a * xx - b * yy) / c


    n = 100
    max_iterations = 100
    goal_inliers = n * 0.3

    # test data
    # xyzs = np.random.random((n, 3)) * 10
    # xyzs[:50, 2:] = xyzs[:50, :1]
    xyzs = bg_xyz_resh

    ax.scatter3D(xyzs.T[0], xyzs.T[1], xyzs.T[2])  # 散点图

    # RANSAC
    m, b = run_ransac(xyzs, estimate, lambda x, y: is_inlier(x, y, 0.01), 3, goal_inliers, max_iterations)
    a, b, c, d = m
    xx, yy, zz = plot_plane(a, b, c, d)
    ax.plot_surface(xx, yy, zz, color=(0, 1, 0, 0.5))

    plt.show()

    '''拟合好平面，测试其他图'''
    depth_test = cv2.imread('data/depth_000170.png', -1)

    xx_test_cam = (gridxx - ox) / fx * depth_test
    yy_test_cam = (gridyy - oy) / fy * depth_test

    # test_xyz = np.concatenate((xx_test_cam[..., np.newaxis], yy_test_cam[..., np.newaxis], depth_test[..., np.newaxis]), axis=2)
    dis = abs(a * xx_test_cam + b * yy_test_cam + c * depth_test + d) / ((a ** 2 + b ** 2 + c ** 2) ** 0.5)
    # dis=np.squeeze(dis)
    depth_test[dis < 50] = 0  # 设定阈值为50mm
    cv2.imshow('', depth_test / 5000)
    cv2.waitKey()
    # cv2.imwrite('data/result_dis.png', dis,-1)
    pass
