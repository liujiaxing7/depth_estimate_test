import argparse
import sys, os
import cv2
import numpy as np

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../'))

from onnx_utils.onnxmodel import ONNXModel
from utils.file_utils import get_left_right_files

from utils.preprocess_postprocess import preprocess_hit, preprocess_madnet
from utils.file_utils import WriteDepth
from utils.file_utils import get_files, get_last_name
from utils.compare_tof import compare_depth_tof
from utils.compare_predict_gt_disp import compare_depth_disp
from utils.compare_tof import get_boundary, get_boundary_wh

def get_parameter():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default=None, required=False, type=str, help='Data directory for prediction')

    parser.add_argument('--model_type', default="madnet", type=str, help='model type of onnx')

    parser.add_argument('--height', default=None, type=int, help='Image height for inference')
    parser.add_argument('--width', default=None, type=int, help='Image width for inference')

    parser.add_argument('--output_dir', default=None, type=str,
                        help='Directory to save inference results and test results')

    parser.add_argument('--onnx_file', default=None, type=str,
                        help='File to save onnx inference model')

    parser.add_argument("--disp_dir", default=None, type=str, help='Dir to save image disp')

    parser.add_argument("--depth_dir", default=None, type=str, help='Dir to save image depth to compare tof')

    parser.add_argument("--tof_dir", default=None, type=str, help="Dir to save images of tof point")

    parser.add_argument("--tof_selected", default=None, type=str, help="Dir to save images of tof points selected")

    parser.add_argument('--bf', type=str, default=None, help='bf for test to generate depth, only parker'
                                                             ' need this parameter now')

    parser.add_argument('--center_crop', type=str, default=None, help='bf for test to generate depth, only parker'
                                                             ' need this parameter now')

    parser.add_argument('--without_tof', action="store_true", default=False, help="indemind data label is not tof")

    parser.add_argument('--scale', type=str, default=100, help="depth image real cm * scale for test depth image")

    return parser.parse_args()
def main():
    args = get_parameter()

    depth_from_onnx = False

    print("current dataset's bf is {}".format(args.bf))

    if args.data_dir is not None and os.path.isdir(args.data_dir):
        left_files, right_files = get_left_right_files(args.data_dir)
        disp_true = [None] * len(left_files)
        if args.disp_dir is not None:
            disp_true = get_files(args.disp_dir)

        # load onnx file
        model = ONNXModel(args.onnx_file)
        depth_from_onnx = True

        root_len = len(args.data_dir)

        for left_file, right_file, disp_file in zip(sorted(left_files), sorted(right_files), sorted(disp_true)):
            if left_file[root_len:][0] == '/':
                op = left_file[root_len + 1:]
            else:
                op = left_file[root_len:]
            left_image = cv2.imread(left_file)
            right_image = cv2.imread(right_file)
            if args.center_crop is not None:
                left, right, top, bottom = get_boundary(left_image, args.center_crop)
                left_image = left_image[top: bottom, left: right]
                right_image = right_image[top: bottom, left: right]
            elif args.width is not None and args.height is not None:
                left, right, top, bottom = get_boundary_wh(left_image, width=int(args.width), height=int(args.height))
                left_image = left_image[top: bottom, left: right]
                right_image = right_image[top: bottom, left: right]

            left_copy = left_image.copy()
            if args.model_type == "madnet":
                left_image = preprocess_madnet(left_image)
                right_image = preprocess_madnet(right_image)
            elif args.model_type == "hitnet":
                left_image = preprocess_hit(left_image)
                right_image = preprocess_hit(right_image)

            output = model.forward2((left_image, right_image))
            if args.model_type == "madnet":
                disp = output[0]
            elif args.model_type == "hitnet":
                disp = output[0][:, 0:1]
                disp = np.clip(disp / 192 * 255, 0, 255)
            if disp_file is not None:
                print("disp_file", disp_file)
                compare_depth_disp(args.output_dir, op, disp, disp_file, bf=args.bf
                                   , center_crop=args.center_crop, without_tof=args.without_tof
                                   ,scale=args.scale, width=args.width, height=args.height)
            op = op.replace(".jpg", ".png")

            WriteDepth(disp, left_copy, args.output_dir, op, args.bf)

            depth_from_onnx = True

    if args.tof_dir is not None:
        tof_lists = get_files(args.tof_dir)
        if args.tof_selected is not None:
            tof_selected_lists = get_files(args.tof_selected)
            for tof_file, tof_selected_file in zip(tof_lists, tof_selected_lists):
                assert get_last_name(tof_file) == get_last_name(tof_selected_file) \
                    , "tof file and tof select is not same"
        else:
            tof_selected_lists = [None] * len(tof_lists)

    if depth_from_onnx and args.depth_dir:
        assert 0, "depth image from model and depth_dir, please input one"
    elif depth_from_onnx:
        depth_files = get_files(os.path.join(args.output_dir, "depth_psl"))
    elif args.depth_dir is not None:
        depth_files = get_files(args.depth_dir)

    if args.tof_dir is not None and len(tof_lists) > 0:
        root_len = len(args.output_dir)
        for tof_file, tof_selected_file, depth_file in zip(tof_lists, tof_selected_lists, depth_files):
            assert get_last_name(tof_file) == get_last_name(depth_file), "tof file: {} and depth file: {} is not same!".format(tof_file, depth_file)

            if depth_file[root_len:][0] == '/':
                op = os.path.join(args.output_dir, depth_file[root_len + 1:])
            else:
                op = os.path.join(args.output_dir, depth_file[root_len:])

            compare_depth_tof(args.output_dir, op, depth_file, tof_file, tof_selected_file, args.center_crop, width=args.width, height=args.height)


    # if args.disp_dir is not None:
    #     disp_true = get_files(args.disp_dir)
    #     for disp_file, depth_file in zip(disp_true, depth_files):
    #         assert get_last_name(disp_file) == get_last_name(
    #             depth_file), "gt disp file: {} and depth file: {} is not same!".format(tof_file, depth_file)
    #         compare_depth_disp(args.output_dir, op, depth_file.replace("depth_psl", "gray"), disp_file, args.bf)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()