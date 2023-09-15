from losses_and_merics import EndPointError, Bad3
import cv2
import tensorflow as tf
import numpy as np
import os
from file_utils import MkdirSimple
import matplotlib.pyplot as plt
from utils.compare_tof import get_boundary, get_boundary_wh

def val_cv2_tf_io(disp_file):

    disp_bytes = tf.io.read_file(disp_file)
    # Using uint16 for higher precision
    disp_map = tf.io.decode_png(disp_bytes, dtype=tf.uint16)
    print("uint16 max: ,min: ",np.max(disp_map), np.min(disp_map))
    disp_map = tf.cast(disp_map, dtype=tf.float32)
    print("float32 max: ,min: ",np.max(disp_map), np.min(disp_map))

    disp_map = disp_map / 256.0

    print("float32/256 max: ,min: ",np.max(disp_map), np.min(disp_map))

    cv_image = cv2.imread(disp_file, -1) /256.0
    print("cv_image max: ,min: ",np.max(cv_image), np.min(cv_image))

    print("sum: {}".format(np.sum(cv_image - disp_map[:,:,0])))

    print(disp_map.shape,cv_image.shape)

def getAbsdiff(depth_map, disparity_map, path, name, bf):
    name = os.path.splitext(name)[0]
    output_depth = os.path.join(path, "diff", name)
    MkdirSimple(output_depth)

    depth_map = np.squeeze(depth_map)

    disparity_map =np.squeeze(disparity_map)# * 256.0

    mask = disparity_map > 0

    # 归一化数据范围
    # depth_map_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map_norm = depth_map
    depth_map_norm_mask = depth_map * mask
    # disparity_map_norm = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX)
    disparity_map_norm = disparity_map * mask
    # print("depth_map_norm: ", depth_map_norm)

    # 计算深度图和视差图的差异
    # diff_map = cv2.absdiff(depth_map_norm_mask.astype("float32"), disparity_map_norm.astype("float32"))
    diff_map = np.abs(bf/depth_map_norm_mask - bf/disparity_map_norm)
    # diff_map = cv2.bitwise_and(diff_map, mask)

    # 计算深度图和视差图的差异
    fig, axes = plt.subplots(4, 1, figsize=(6, 12))
    # axes[0].subplot(1, 3, 1)
    axes[0].imshow(depth_map_norm, cmap='gray')
    axes[0].set_title('Depth Map')
    axes[0].axis('off')
    cv2.imwrite(os.path.join(output_depth, "depth_map_norm.png"), depth_map_norm)

    axes[1].imshow(depth_map_norm_mask, cmap='gray')
    axes[1].set_title('Depth Map Mask')
    axes[1].axis('off')
    cv2.imwrite(os.path.join(output_depth, "depth_map_norm_mask.png"), depth_map_norm_mask)

    # axes[1].subplot(1, 3, 2)
    axes[2].imshow(disparity_map_norm, cmap='gray')
    axes[2].set_title('Disparity Map')
    axes[2].axis('off')
    cv2.imwrite(os.path.join(output_depth, "disparity_map_norm.png"), disparity_map_norm)

    # axes[2].subplot(1, 3, 3)
    cax = axes[3].imshow(diff_map, cmap='jet')
    axes[3].set_title('Difference Map')
    # axes[2].colorbar()
    axes[3].axis('off')
    diff_map1 = diff_map.copy()
    # diff_map = cv2.applyColorMap(np.asarray(diff_map).astype("uint8"), cv2.COLORMAP_JET)
    output_disp_name = os.path.join("./result1", name + ".png")
    if not os.path.exists(os.path.split(output_disp_name)[0]):
        os.makedirs(os.path.split(output_disp_name)[0])
    cv2.imwrite(output_disp_name, diff_map.astype("uint16"))
    # diff_map1 =  cv2.applyColorMap(np.asarray(diff_map1).astype("uint8") * 5, cv2.COLORMAP_JET)
    # cv2.imwrite(os.path.join(output_depth, "diff_map_scale_5.png"), diff_map1)
    #
    # concat = np.vstack([cv2.cvtColor(depth_map_norm, cv2.COLOR_GRAY2BGR), cv2.cvtColor(depth_map_norm_mask, cv2.COLOR_GRAY2BGR)
    #                        ,cv2.cvtColor(disparity_map_norm, cv2.COLOR_GRAY2BGR), diff_map])
    # cv2.imwrite(os.path.join(output_depth, "concat_compare_map.png"), concat)
    #
    # fig.subplots_adjust(hspace=0.05, wspace=0.05, top=0.95, bottom=0.05)
    # plt.colorbar(cax, ax=axes[3])

    # 调整图像尺寸
    # fig.set_size_inches(10, 15)
    #
    # plt.tight_layout()
    # plt.show()
    # plt.savefig("myplot.png")


def compare_depth_disp(output_dir, op, disp_output, disp_file, bf=None, center_crop=None, without_tof=False
                       , scale=100.0, width=None, height=None):
    gt = cv2.imread(disp_file, -1)
    gt = gt /256.0
    if bf is not None:
        gt = cv2.imread(disp_file, -1)
        if not without_tof:
            gt = gt[:, :, 0] + (gt[:, :, 1] > 0) * 255 + gt[:, :, 1] + (
                        gt[:, :, 2] > 0) * 511 + gt[:, :, 2]
        else:
            gt = gt / int(scale)
            if (gt.shape[-1] ==3):
                gt = gt[:,:,-1]
            elif gt.shape[-1] ==1:
                pass
            else:
                print("read depth(disp) image error, channels is wrong: {}".format(disp_file))
        if center_crop is not None:
            left, right, top, bottom = get_boundary(gt, center_crop)
            gt = gt[top: bottom, left: right]
        elif width is not None and height is not None:
            left, right, top, bottom = get_boundary_wh(gt, width=width, height=height)
            gt = gt[top: bottom, left: right]

        bf = float(bf)
        mask = gt > 0
        gt[mask] = bf / gt[mask]
        print("gt max: ", np.max(gt), np.min(gt))
        print(gt.shape)
    elif width is not None and height is not None:
        left, right, top, bottom = get_boundary_wh(gt, width=width, height=height)
        gt = gt[top: bottom, left: right]

    disparities = disp_output

    epe = EndPointError()
    gt_flaot = gt.astype(np.float32)
    disparities_float = np.squeeze(disparities).astype(np.float32)
    epe.update_state(gt_flaot, disparities_float)
    print("EPE: ", epe.result())

    bad3 = Bad3()
    bad3.update_state(gt_flaot, disparities_float)
    print("Bad3: ", bad3.result())

    getAbsdiff(disparities_float, gt_flaot, output_dir, op, bf)

if __name__ == '__main__':
    # compare_depth_disp("test_c","/home/indemind/Code/PycharmProjects/Depth_Estimation/Stereo/madnet/result_D10.0.7_2000_test/gray/000000_10.png","/home/indemind/Code/PycharmProjects/Depth_Estimation/Stereo/madnet/result_D10.0.7_2000_test/gray/000000_10.png","/home/indemind/Code/PycharmProjects/Depth_Estimation/Stereo/madnet/test_images/kitti/disp/000000_10.png")
    disp_output = cv2.imread("/home/indemind/Code/PycharmProjects/Depth_Estimation/Stereo/madnet/result_D10.0.7_2000_test/gray/000000_10.png", -1)
    compare_depth_disp("test_c",1, disp_output, "/home/indemind/Code/PycharmProjects/Depth_Estimation/Stereo/madnet/test_images/kitti/disp/000000_10.png")