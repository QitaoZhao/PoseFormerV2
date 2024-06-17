import os
import numpy as np
from common.utils_3dhp import *

import scipy.io as scio

data_path=r'F:\mpi_inf_3dhp\data'
cam_set = [0, 1, 2, 4, 5, 6, 7, 8]
# joint_set = [8, 6, 15, 16, 17, 10, 11, 12, 24, 25, 26, 19, 20, 21, 5, 4, 7]
joint_set = [7, 5, 14, 15, 16, 9, 10, 11, 23, 24, 25, 18, 19, 20, 4, 3, 6]

dic_seq={}

for root, dirs, files in os.walk(data_path):

    for file in files:
        if file.endswith("mat"):

            path = root.split("\\")
            subject = path[-2][1]
            seq = path[-1][3]
            print("loading %s %s..."%(path[-2],path[-1]))

            temp = mpii_get_sequence_info(subject, seq)

            frames = temp[0]
            fps = temp[1]

            data = scio.loadmat(os.path.join(root, file))
            cameras = data['cameras'][0]
            for cam_idx in range(len(cameras)):
                assert cameras[cam_idx] == cam_idx

            data_2d = data['annot2'][cam_set]
            data_3d = data['univ_annot3'][cam_set]

            dic_cam = {}
            a  = len(data_2d)
            for cam_idx in range(len(data_2d)):
                data_2d_cam = data_2d[cam_idx][0]
                data_3d_cam = data_3d[cam_idx][0]

                data_2d_cam = data_2d_cam.reshape(data_2d_cam.shape[0], 28,2)
                data_3d_cam = data_3d_cam.reshape(data_3d_cam.shape[0], 28,3)

                data_2d_select = data_2d_cam[:frames, joint_set]
                data_3d_select = data_3d_cam[:frames, joint_set]

                dic_data = {"data_2d":data_2d_select,"data_3d":data_3d_select}

                dic_cam.update({str(cam_set[cam_idx]):dic_data})


            dic_seq.update({path[-2]+" "+path[-1]:[dic_cam, fps]})


np.savez_compressed('data_train_3dhp', data=dic_seq)









