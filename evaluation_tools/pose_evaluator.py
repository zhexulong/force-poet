# ------------------------------------------------------------------------
# PoET: Pose Estimation Transformer for Single-View, Multi-Object 6D Pose Estimation
# Copyright (c) 2022 Thomas Jantos (thomas.jantos@aau.at), University of Klagenfurt - Control of Networked Systems (CNS). All Rights Reserved.
# Licensed under the BSD-2-Clause-License with no commercial use [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from CDPN (https://github.com/LZGMatrix/CDPN_ICCV2019_ZhigangLi)
# Licensed under the Apache License, Version 2.0 [see LICENSE_CDPN in the LICENSES folder for details]
# ------------------------------------------------------------------------
from __future__ import print_function, division, absolute_import

import os
import shutil
import copy
import json

from scipy import spatial
import numpy as np
from scipy.linalg import logm
import numpy.linalg as LA


class PoseEvaluator(object):
    def __init__(self, models, classes, model_info, model_symmetry, depth_scale=0.1):
        """
        Initialization of the Pose Evaluator for the YCB-V dataset.

        It can calculate the average rotation and translation error, as well as the ADD, ADD-S and ADD-(S) metric
        as described in https://arxiv.org/pdf/1711.00199.pdf (PoseCNN)

        Parameters
            - models: Array containing the points of each object 3D model (Contains the 3D points for each class)
            - classes: Array containing the information about the object classes (mapping between class ids and class names)
            - model_info: Information about the models (diameter and extension)
            - model_symmetry: Indication whether the 3D model of a certain class is symmetric (axis, plane) or not.
        """
        self.models = models
        self.classes = [classes[k] for k in classes]
        self.classes_map = classes
        self.models_info = model_info
        self.model_symmetry = model_symmetry

        self.poses_pred = {}
        self.poses_gt = {}
        self.poses_img = {}
        self.camera_intrinsics = {}
        self.num = {}
        
        # Add force evaluation storage
        self.forces_pred = {}
        self.forces_gt = {}
        self.force_errors = {}
        
        # Add force matrix evaluation storage
        self.force_matrices_pred = {}
        self.force_matrices_gt = {}
        self.force_matrix_errors = {}
        
        self.depth_scale = depth_scale

        self.reset()  # Initialize

        self.writer = None
        self.training = False
        self.testing = False

    def reset(self):
        """
        Reset the PoseEvaluator stored poses. Necessary when the same evaluator is used during training
        """
        self.poses_pred = {}
        self.poses_gt = {}
        self.poses_img = {}
        self.camera_intrinsics = {}
        self.num = {}
        
        # Reset force evaluation storage
        self.forces_pred = {}
        self.forces_gt = {}
        self.force_errors = {}
        
        # Reset force matrix evaluation storage
        self.force_matrices_pred = {}
        self.force_matrices_gt = {}
        self.force_matrix_errors = {}

        for cls in self.classes:
            self.num[cls] = 0.
            self.poses_pred[cls] = []
            self.poses_gt[cls] = []
            self.poses_img[cls] = []
            self.camera_intrinsics[cls] = []
            
            # Initialize force storage for each class
            self.forces_pred[cls] = []
            self.forces_gt[cls] = []
            self.force_errors[cls] = []
            
            # Initialize force matrix storage for each class
            self.force_matrices_pred[cls] = []
            self.force_matrices_gt[cls] = []
            self.force_matrix_errors[cls] = []

    def evaluate_pose_adds(self, output_path):
        """
        Evaluate 6D pose by ADD(-S) metric
        Symmetric Object --> ADD-S
        NonSymmetric Objects --> ADD

        For metric definition we refer to PoseCNN: https://arxiv.org/pdf/1711.00199.pdf
        """
        output_dir = output_path + "adds/"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        log_file = open(output_path + "adds/adds.log", 'w')
        json_file = open(output_path + "adds/adds.json", 'w')

        poses_pred = self.poses_pred
        poses_gt = self.poses_gt
        models = self.models
        model_symmetry = self.model_symmetry

        log_file.write('\n* {} *\n {:^}\n* {} *'.format('-' * 100, 'Metric ADD(-S)', '-' * 100))
        log_file.write("\n")

        n_classes = len(self.classes)
        count_all = np.zeros((n_classes), dtype=np.float32)
        count_correct = {k: np.zeros((n_classes), dtype=np.float32) for k in ['0.02', '0.05', '0.10']}

        threshold_002 = np.zeros((n_classes), dtype=np.float32)
        threshold_005 = np.zeros((n_classes), dtype=np.float32)
        threshold_010 = np.zeros((n_classes), dtype=np.float32)
        dx = 0.0001
        threshold_mean = np.tile(np.arange(0, 0.1, dx).astype(np.float32), (n_classes, 1))  # (num_class, num_thresh)
        num_thresh = threshold_mean.shape[-1]
        count_correct['mean'] = np.zeros((n_classes, num_thresh), dtype=np.float32)

        adds_results = {}
        adds_results["thresholds"] = [0.02, 0.05, 0.10]

        self.classes = sorted(self.classes)
        num_valid_class = len(self.classes)
        for i, cls_name in enumerate(self.classes):
            threshold_002[i] = 0.02
            threshold_005[i] = 0.05
            threshold_010[i] = 0.10

            symmetry_flag = model_symmetry[cls_name]
            cls_poses_pred = poses_pred[cls_name]
            cls_poses_gt = poses_gt[cls_name]
            model_pts = models[cls_name]['pts']
            n_poses = len(cls_poses_gt)
            count_all[i] = n_poses
            for j in range(n_poses):
                pose_pred = cls_poses_pred[j]  # est pose
                pose_gt = cls_poses_gt[j]  # gt pose
                if symmetry_flag:
                    eval_method = 'adi'
                    error = self.calc_adi(model_pts, pose_pred, pose_gt)
                else:
                    eval_method = 'add'
                    error = self.calc_add(model_pts, pose_pred, pose_gt)
                if error < threshold_002[i]:
                    count_correct['0.02'][i] += 1
                if error < threshold_005[i]:
                    count_correct['0.05'][i] += 1
                if error < threshold_010[i]:
                    count_correct['0.10'][i] += 1
                for thresh_i in range(num_thresh):
                    if error < threshold_mean[i, thresh_i]:
                        count_correct['mean'][i, thresh_i] += 1
            adds_results[cls_name] = {}
            adds_results[cls_name]["threshold"] = {'0.02': count_correct['0.02'][i].tolist(),
                                                   '0.05': count_correct['0.05'][i].tolist(),
                                                   '0.10': count_correct['0.10'][i].tolist(),
                                                   'mean': count_correct['mean'][i].tolist()}

        plot_data = {}
        sum_acc_mean = np.zeros(1)
        sum_acc_002 = np.zeros(1)
        sum_acc_005 = np.zeros(1)
        sum_acc_010 = np.zeros(1)
        for i, cls_name in enumerate(self.classes):
            if count_all[i] == 0:
                continue
            plot_data[cls_name] = []
            log_file.write("** {} **".format(cls_name))
            from scipy.integrate import simps
            area = simps(count_correct['mean'][i] / float(count_all[i]), dx=dx) / 0.1
            acc_mean = area * 100
            sum_acc_mean[0] += acc_mean
            acc_002 = 100 * float(count_correct['0.02'][i]) / float(count_all[i])
            sum_acc_002[0] += acc_002
            acc_005 = 100 * float(count_correct['0.05'][i]) / float(count_all[i])
            sum_acc_005[0] += acc_005
            acc_010 = 100 * float(count_correct['0.10'][i]) / float(count_all[i])
            sum_acc_010[0] += acc_010

            log_file.write('threshold=[0.0, 0.10], area: {:.2f}'.format(acc_mean))
            log_file.write("\n")
            log_file.write('threshold=0.02, correct poses: {}, all poses: {}, accuracy: {:.2f}'.format(
                count_correct['0.02'][i],
                count_all[i],
                acc_002))
            log_file.write("\n")
            log_file.write('threshold=0.05, correct poses: {}, all poses: {}, accuracy: {:.2f}'.format(
                count_correct['0.05'][i],
                count_all[i],
                acc_005))
            log_file.write("\n")
            log_file.write('threshold=0.10, correct poses: {}, all poses: {}, accuracy: {:.2f}'.format(
                count_correct['0.10'][i],
                count_all[i],
                acc_010))
            log_file.write("\n")
            log_file.write("\n")
            adds_results[cls_name]["accuracy"] = {'n_poses': count_all[i].tolist(),
                                                  '0.02': acc_002,
                                                  '0.05': acc_005,
                                                  '0.10': acc_010,
                                                  'auc': acc_mean}

        log_file.write("=" * 30)
        log_file.write("\n")

        for iter_i in range(1):
            log_file.write("---------- ADD(-S) performance over {} classes -----------".format(num_valid_class))
            log_file.write("\n")
            log_file.write("** iter {} **".format(iter_i + 1))
            log_file.write("\n")
            log_file.write('threshold=[0.0, 0.10], area: {:.2f}'.format(
                sum_acc_mean[iter_i] / num_valid_class))
            log_file.write("\n")
            log_file.write('threshold=0.02, mean accuracy: {:.2f}'.format(
                sum_acc_002[iter_i] / num_valid_class))
            log_file.write("\n")
            log_file.write('threshold=0.05, mean accuracy: {:.2f}'.format(
                sum_acc_005[iter_i] / num_valid_class))
            log_file.write("\n")
            log_file.write('threshold=0.10, mean accuracy: {:.2f}'.format(
                sum_acc_010[iter_i] / num_valid_class))
            log_file.write("\n")
        log_file.write("=" * 30)
        adds_results["accuracy"] = {'0.02': sum_acc_002[0].tolist() / num_valid_class,
                                    '0.05': sum_acc_005[0].tolist() / num_valid_class,
                                    '0.10': sum_acc_010[0].tolist() / num_valid_class,
                                    'auc': sum_acc_mean[0].tolist() / num_valid_class}

        log_file.write("\n")
        log_file.close()
        json.dump(adds_results, json_file)
        json_file.close()
        return

    def evaluate_pose_adi(self, output_path):
        """
        Evaluate 6D pose by ADD-S metric

        For metric definition we refer to PoseCNN: https://arxiv.org/pdf/1711.00199.pdf
        """
        output_dir = output_path + "adi/"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        log_file = open(output_path + "adi/adds.log", 'w')
        json_file = open(output_path + "adi/adds.json", 'w')

        poses_pred = copy.deepcopy(self.poses_pred)
        poses_gt = copy.deepcopy(self.poses_gt)
        models = self.models

        log_file.write('\n* {} *\n {:^}\n* {} *'.format('-' * 100, 'Metric ADD-S', '-' * 100))
        log_file.write("\n")

        eval_method = 'adi'
        n_classes = len(self.classes)
        count_all = np.zeros((n_classes), dtype=np.float32)
        count_correct = {k: np.zeros((n_classes), dtype=np.float32) for k in ['0.02', '0.05', '0.10']}

        threshold_002 = np.zeros((n_classes), dtype=np.float32)
        threshold_005 = np.zeros((n_classes), dtype=np.float32)
        threshold_010 = np.zeros((n_classes), dtype=np.float32)
        dx = 0.0001
        threshold_mean = np.tile(np.arange(0, 0.1, dx).astype(np.float32), (n_classes, 1))  # (num_class, num_thresh)
        num_thresh = threshold_mean.shape[-1]
        count_correct['mean'] = np.zeros((n_classes, num_thresh), dtype=np.float32)

        adi_results = {}
        adi_results["thresholds"] = [0.02, 0.05, 0.10]

        self.classes = sorted(self.classes)
        num_valid_class = len(self.classes)
        for i, cls_name in enumerate(self.classes):
            threshold_002[i] = 0.02
            threshold_005[i] = 0.05
            threshold_010[i] = 0.10

            cls_poses_pred = poses_pred[cls_name]
            cls_poses_gt = poses_gt[cls_name]
            model_pts = models[cls_name]['pts']
            n_poses = len(cls_poses_gt)
            count_all[i] = n_poses
            for j in range(n_poses):
                pose_pred = cls_poses_pred[j]  # est pose
                pose_gt = cls_poses_gt[j]  # gt pose
                error = self.calc_adi(model_pts, pose_pred, pose_gt)
                if error < threshold_002[i]:
                    count_correct['0.02'][i] += 1
                if error < threshold_005[i]:
                    count_correct['0.05'][i] += 1
                if error < threshold_010[i]:
                    count_correct['0.10'][i] += 1
                for thresh_i in range(num_thresh):
                    if error < threshold_mean[i, thresh_i]:
                        count_correct['mean'][i, thresh_i] += 1
            adi_results[cls_name] = {}
            adi_results[cls_name]["threshold"] = {'0.02': count_correct['0.02'][i].tolist(),
                                                   '0.05': count_correct['0.05'][i].tolist(),
                                                   '0.10': count_correct['0.10'][i].tolist(),
                                                   'mean': count_correct['mean'][i].tolist()}

        plot_data = {}
        sum_acc_mean = np.zeros(1)
        sum_acc_002 = np.zeros(1)
        sum_acc_005 = np.zeros(1)
        sum_acc_010 = np.zeros(1)
        for i, cls_name in enumerate(self.classes):
            if count_all[i] == 0:
                continue
            plot_data[cls_name] = []
            log_file.write("** {} **".format(cls_name))
            from scipy.integrate import simps
            area = simps(count_correct['mean'][i] / float(count_all[i]), dx=dx) / 0.1
            acc_mean = area * 100
            sum_acc_mean[0] += acc_mean
            acc_002 = 100 * float(count_correct['0.02'][i]) / float(count_all[i])
            sum_acc_002[0] += acc_002
            acc_005 = 100 * float(count_correct['0.05'][i]) / float(count_all[i])
            sum_acc_005[0] += acc_005
            acc_010 = 100 * float(count_correct['0.10'][i]) / float(count_all[i])
            sum_acc_010[0] += acc_010

            log_file.write('threshold=[0.0, 0.10], area: {:.2f}'.format(acc_mean))
            log_file.write("\n")
            log_file.write('threshold=0.02, correct poses: {}, all poses: {}, accuracy: {:.2f}'.format(
                count_correct['0.02'][i],
                count_all[i],
                acc_002))
            log_file.write("\n")
            log_file.write('threshold=0.05, correct poses: {}, all poses: {}, accuracy: {:.2f}'.format(
                count_correct['0.05'][i],
                count_all[i],
                acc_005))
            log_file.write("\n")
            log_file.write('threshold=0.10, correct poses: {}, all poses: {}, accuracy: {:.2f}'.format(
                count_correct['0.10'][i],
                count_all[i],
                acc_010))
            log_file.write("\n")
            log_file.write("\n")

            adi_results[cls_name]["accuracy"] = {'n_poses': count_all[i].tolist(),
                                                 '0.02': acc_002,
                                                 '0.05': acc_005,
                                                 '0.10': acc_010,
                                                 'auc': acc_mean}

        log_file.write("=" * 30)
        log_file.write('\n')

        for iter_i in range(1):
            log_file.write("---------- ADD-S performance over {} classes -----------".format(num_valid_class))
            log_file.write("\n")
            log_file.write("** iter {} **".format(iter_i + 1))
            log_file.write("\n")
            log_file.write('threshold=[0.0, 0.10], area: {:.2f}'.format(
                sum_acc_mean[iter_i] / num_valid_class))
            log_file.write("\n")
            log_file.write('threshold=0.02, mean accuracy: {:.2f}'.format(
                sum_acc_002[iter_i] / num_valid_class))
            log_file.write("\n")
            log_file.write('threshold=0.05, mean accuracy: {:.2f}'.format(
                sum_acc_005[iter_i] / num_valid_class))
            log_file.write("\n")
            log_file.write('threshold=0.10, mean accuracy: {:.2f}'.format(
                sum_acc_010[iter_i] / num_valid_class))
            log_file.write("\n")
        log_file.write("=" * 30)
        adi_results["accuracy"] = {'0.02': sum_acc_002[0].tolist() / num_valid_class,
                                    '0.05': sum_acc_005[0].tolist() / num_valid_class,
                                    '0.10': sum_acc_010[0].tolist() / num_valid_class,
                                    'auc': sum_acc_mean[0].tolist() / num_valid_class}

        log_file.write("\n")
        log_file.close()
        json.dump(adi_results, json_file)
        json_file.close()
        return

    def evaluate_pose_add(self, output_path):
        """
        Evaluate 6D pose by ADD Metric

        For metric definition we refer to PoseCNN: https://arxiv.org/pdf/1711.00199.pdf
        """

        output_dir = output_path + "/add/"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        log_file = open(output_path + "add/add.log", 'w')
        json_file = open(output_path + "add/add.json", 'w')

        poses_pred = copy.deepcopy(self.poses_pred)
        poses_gt = copy.deepcopy(self.poses_gt)
        models_info = self.models_info
        models = self.models

        log_file.write('\n* {} *\n {:^}\n* {} *'.format('-' * 100, 'Metric ADD', '-' * 100))
        log_file.write("\n")

        eval_method = 'add'
        n_classes = len(self.classes)
        count_all = np.zeros((n_classes), dtype=np.float32)
        count_correct = {k: np.zeros((n_classes), dtype=np.float32) for k in ['0.02', '0.05', '0.10']}

        threshold_002 = np.zeros((n_classes), dtype=np.float32)
        threshold_005 = np.zeros((n_classes), dtype=np.float32)
        threshold_010 = np.zeros((n_classes), dtype=np.float32)
        dx = 0.0001
        threshold_mean = np.tile(np.arange(0, 0.1, dx).astype(np.float32), (n_classes, 1)) # (num_class, num_thresh)
        num_thresh = threshold_mean.shape[-1]
        count_correct['mean'] = np.zeros((n_classes, num_thresh), dtype=np.float32)

        add_results = {}
        add_results["thresholds"] = [0.02, 0.05, 0.10]

        self.classes = sorted(self.classes)
        num_valid_class = len(self.classes)
        for i, cls_name in enumerate(self.classes):
            threshold_002[i] = 0.02
            threshold_005[i] = 0.05
            threshold_010[i] = 0.10
            # threshold_mean[i, :] *= models_info[cls_name]['diameter']
            cls_poses_pred = poses_pred[cls_name]
            cls_poses_gt = poses_gt[cls_name]
            model_pts = models[cls_name]['pts']
            n_poses = len(cls_poses_gt)
            count_all[i] = n_poses
            for j in range(n_poses):
                pose_pred = cls_poses_pred[j]  # est pose
                pose_gt = cls_poses_gt[j]  # gt pose
                error = self.calc_add(model_pts, pose_pred, pose_gt)
                if error < threshold_002[i]:
                    count_correct['0.02'][i] += 1
                if error < threshold_005[i]:
                    count_correct['0.05'][i] += 1
                if error < threshold_010[i]:
                    count_correct['0.10'][i] += 1
                for thresh_i in range(num_thresh):
                    if error < threshold_mean[i, thresh_i]:
                        count_correct['mean'][i, thresh_i] += 1
            add_results[cls_name] = {}
            add_results[cls_name]["threshold"] = {'0.02': count_correct['0.02'][i].tolist(),
                                                   '0.05': count_correct['0.05'][i].tolist(),
                                                   '0.10': count_correct['0.10'][i].tolist(),
                                                   'mean': count_correct['mean'][i].tolist()}

        plot_data = {}
        sum_acc_mean = np.zeros(1)
        sum_acc_002 = np.zeros(1)
        sum_acc_005 = np.zeros(1)
        sum_acc_010 = np.zeros(1)
        for i, cls_name in enumerate(self.classes):
            if count_all[i] == 0:
                continue
            plot_data[cls_name] = []
            log_file.write("** {} **".format(cls_name))
            from scipy.integrate import simps
            area = simps(count_correct['mean'][i] / float(count_all[i]), dx=dx) / 0.1
            acc_mean = area * 100
            sum_acc_mean[0] += acc_mean
            acc_002 = 100 * float(count_correct['0.02'][i]) / float(count_all[i])
            sum_acc_002[0] += acc_002
            acc_005 = 100 * float(count_correct['0.05'][i]) / float(count_all[i])
            sum_acc_005[0] += acc_005
            acc_010 = 100 * float(count_correct['0.10'][i]) / float(count_all[i])
            sum_acc_010[0] += acc_010

            log_file.write('threshold=[0.0, 0.10], area: {:.2f}'.format(acc_mean))
            log_file.write("\n")
            log_file.write('threshold=0.02, correct poses: {}, all poses: {}, accuracy: {:.2f}'.format(
                count_correct['0.02'][i],
                count_all[i],
                acc_002))
            log_file.write("\n")
            log_file.write('threshold=0.05, correct poses: {}, all poses: {}, accuracy: {:.2f}'.format(
                count_correct['0.05'][i],
                count_all[i],
                acc_005))
            log_file.write("\n")
            log_file.write('threshold=0.10, correct poses: {}, all poses: {}, accuracy: {:.2f}'.format(
                count_correct['0.10'][i],
                count_all[i],
                acc_010))
            log_file.write("\n")
            log_file.write("\n")
            add_results[cls_name]["accuracy"] = {'n_poses': count_all[i].tolist(),
                                                  '0.02': acc_002,
                                                  '0.05': acc_005,
                                                  '0.10': acc_010,
                                                  'auc': acc_mean}

        log_file.write("=" * 30)
        log_file.write("\n")

        for iter_i in range(1):
            log_file.write("---------- ADD performance over {} classes -----------".format(num_valid_class))
            log_file.write("\n")
            log_file.write("** iter {} **".format(iter_i + 1))
            log_file.write("\n")
            log_file.write('threshold=[0.0, 0.10], area: {:.2f}'.format(
                sum_acc_mean[iter_i] / num_valid_class))
            log_file.write("\n")
            log_file.write('threshold=0.02, mean accuracy: {:.2f}'.format(
                sum_acc_002[iter_i] / num_valid_class))
            log_file.write("\n")
            log_file.write('threshold=0.05, mean accuracy: {:.2f}'.format(
                sum_acc_005[iter_i] / num_valid_class))
            log_file.write("\n")
            log_file.write('threshold=0.10, mean accuracy: {:.2f}'.format(
                sum_acc_010[iter_i] / num_valid_class))
            log_file.write("\n")
        log_file.write("=" * 30)

        add_results["accuracy"] = {'0.02': sum_acc_002[0].tolist() / num_valid_class,
                                    '0.05': sum_acc_005[0].tolist() / num_valid_class,
                                    '0.10': sum_acc_010[0].tolist() / num_valid_class,
                                    'auc': sum_acc_mean[0].tolist() / num_valid_class}

        log_file.write("\n")
        log_file.close()
        json.dump(add_results, json_file)
        json_file.close()
        return

    def calculate_class_avg_translation_error(self, output_path, epoch):
        """
        Calculate the average translation error for each class and then the average error across all classes in meters
        """
        output_dir = output_path + "/avg_t_error/"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        log_file = open(output_path + "/avg_t_error/avg_t_error.log", 'w')
        json_file = open(output_path + "avg_t_error/avg_t_error.json", 'w')

        log_file.write('\n* {} *\n {:^}\n* {} *'.format('-' * 100, 'Metric Average Translation Error in Meters', '-' * 100))
        log_file.write("\n")

        poses_pred = self.poses_pred
        poses_gt = self.poses_gt
        translation_errors = []
        cls_translation_errors = {}
        avg_translation_errors = {}
        for cls in self.classes:
            cls_translation_errors[cls] = []
            cls_poses_pred = poses_pred[cls]
            cls_poses_gt = poses_gt[cls]
            for pose_est, pose_gt in zip(cls_poses_pred, cls_poses_gt):
                t_est = pose_est[:, 3]
                t_gt = pose_gt[:, 3]
                error = np.sqrt(np.sum(np.square((t_est - t_gt))))
                cls_translation_errors[cls].append(error)
                translation_errors.append(error)
            if len(cls_translation_errors[cls]) != 0:
                avg_error = np.sum(cls_translation_errors[cls]) / len(cls_translation_errors[cls])
                avg_translation_errors[cls] = avg_error
            else:
                avg_translation_errors[cls] = np.nan
            log_file.write("Class: {} \t\t {}".format(cls, avg_translation_errors[cls]))
            log_file.write("\n")
            if self.writer is not None and self.training == True:
                if not self.testing:
                    self.writer.add_scalar(f'ValClassTrans/avg_trans_err/{cls}', avg_translation_errors[cls], epoch)
                else:
                    self.writer.add_scalar(f'TestClassTrans/avg_trans_err/{cls}', avg_translation_errors[cls])

        total_avg_error = np.sum(translation_errors) / len(translation_errors)
        log_file.write("All:\t\t\t\t\t {}".format(total_avg_error))
        avg_translation_errors["mean"] = [total_avg_error]

        log_file.write("\n")
        log_file.close()
        json.dump(avg_translation_errors, json_file)
        json_file.close()
        return total_avg_error

    def calculate_class_avg_rotation_error(self, output_path, epoch):
        """
        Calculate the average rotation error given by the Geodesic distance for each class and then the average error
        across all classes in degree
        """
        output_dir = output_path + "/avg_rot_error/"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        log_file = open(output_path + "/avg_rot_error/avg_rot_error.log", 'w')
        json_file = open(output_path + "avg_rot_error/avg_rot_error.json", 'w')

        log_file.write(
            '\n* {} *\n {:^}\n* {} *'.format('-' * 100, 'Metric Average Rotation Error in Degrees', '-' * 100))
        log_file.write("\n")

        poses_pred = copy.deepcopy(self.poses_pred)
        poses_gt = copy.deepcopy(self.poses_gt)
        rotation_errors = []
        cls_rotation_errors = {}
        avg_rotation_errors = {}

        for cls in self.classes:
            cls_rotation_errors[cls] = []
            cls_pose_pred = poses_pred[cls]
            cls_pose_gt = poses_gt[cls]
            for pose_est, pose_gt in zip(cls_pose_pred, cls_pose_gt):
                rot_est = pose_est[:3, :3]
                rot_gt = pose_gt[:3, :3]
                rot = np.matmul(rot_est, rot_gt.T)
                trace = np.trace(rot)
                if trace < -1.0:
                    trace = -1
                elif trace > 3.0:
                    trace = 3.0
                angle_diff = np.degrees(np.arccos(0.5 * (trace - 1)))
                cls_rotation_errors[cls].append(angle_diff)
                rotation_errors.append(angle_diff)
            if len(cls_rotation_errors[cls]) != 0:
                avg_error = np.sum(cls_rotation_errors[cls]) / len(cls_rotation_errors[cls])
                avg_rotation_errors[cls] = avg_error
            else:
                avg_rotation_errors[cls] = np.nan
            log_file.write("Class: {} \t\t {}".format(cls, avg_rotation_errors[cls]))
            log_file.write("\n")
            if self.writer is not None and self.training == True:
                if not self.testing:
                    self.writer.add_scalar(f'ValClassRot/avg_rot_err/{cls}', avg_rotation_errors[cls], epoch)
                else:
                    self.writer.add_scalar(f"TestClassRot/avg_rot_err/{cls}", avg_rotation_errors[cls])

        total_avg_error = np.sum(rotation_errors) / len(rotation_errors)
        log_file.write("All:\t\t\t\t\t {}".format(total_avg_error))
        avg_rotation_errors["mean"] = [total_avg_error]

        log_file.write("\n")
        log_file.close()
        json.dump(avg_rotation_errors, json_file)
        json_file.close()
        return total_avg_error

    def se3_mul(self, RT1, RT2):
        """
        Concat 2 RT transform
        :param RT1=[R,T], 4x3 np array
        :param RT2=[R,T], 4x3 np array
        :return: RT_new = RT1 * RT2
        """
        R1 = RT1[0:3, 0:3]
        T1 = RT1[0:3, 3].reshape((3, 1))

        R2 = RT2[0:3, 0:3]
        T2 = RT2[0:3, 3].reshape((3, 1))

        RT_new = np.zeros((3, 4), dtype=np.float32)
        RT_new[0:3, 0:3] = np.dot(R1, R2)
        T_new = np.dot(R1, T2) + T1
        RT_new[0:3, 3] = T_new.reshape((3))
        return RT_new

    def transform_pts(self, pts, rot, t):
        """
        Applies a rigid transformation to 3D points.

        :param pts: nx3 ndarray with 3D points.
        :param rot: 3x3 rotation matrix.
        :param t: 3x1 translation vector.
        :return: nx3 ndarray with transformed 3D points.
        """
        assert (pts.shape[1] == 3)
        pts_t = rot.dot(pts.T) + t.reshape((3, 1))
        return pts_t.T

    def project_pts(self, pts, rot, t, K):
        """
        Applies a rigid transformation to 3D points.

        :param pts: nx3 ndarray with 3D points.
        :param rot: 3x3 rotation matrix.
        :param t: 3x1 translation vector.
        :param K: 3x3 intrinsic matrix
        :return: nx2 ndarray with transformed 2D points.
        """
        assert (pts.shape[1] == 3)
        if K.shape == (9,):
            K = K.reshape(3, 3)
        pts_t = rot.dot(pts.T) + t.reshape((3, 1))  # 3xn
        pts_c_t = K.dot(pts_t)
        n = pts.shape[0]
        pts_2d = np.zeros((n, 2))
        pts_2d[:, 0] = pts_c_t[0, :] / pts_c_t[2, :]
        pts_2d[:, 1] = pts_c_t[1, :] / pts_c_t[2, :]

        return pts_2d

    def proj(self, pts, pose_pred, pose_gt, K):
        '''
        average re-projection error in 2d

        :param pts: nx3 ndarray with 3D model points.
        :param pose_pred: Estimated pose (3x3 rot. matrix and 3x1 translation vector).
        :param pose_gt: GT pose (3x3 rot. matrix and 3x1 translation vector).
        :param K: Camera intrinsics to project the model onto the image plane.
        :return:
        '''
        rot_pred = pose_pred[:3, :3]
        t_pred = pose_pred[:, 3]

        rot_gt = pose_gt[:3, :3]
        t_gt = pose_gt[:, 3]

        proj_pred = self.project_pts(pts, rot_pred, t_pred, K)
        proj_gt = self.project_pts(pts, rot_gt, t_gt, K)
        e = np.linalg.norm(proj_pred - proj_gt, axis=1).mean()
        return e

    def calc_add(self, pts, pose_pred, pose_gt):
        """
        Average Distance of Model Points for objects with no indistinguishable views
        - by Hinterstoisser et al. (ACCV 2012).
        http://www.stefan-hinterstoisser.com/papers/hinterstoisser2012accv.pdf

        :param pts: nx3 ndarray with 3D model points.
        :param pose_pred: Estimated pose (3x3 rot. matrix and 3x1 translation vector).
        :param pose_gt: GT pose (3x3 rot. matrix and 3x1 translation vector).
        :return: Mean average error between the predicted and ground truth pose.
        """
        rot_pred = pose_pred[:3, :3]
        t_pred = pose_pred[:, 3]

        rot_gt = pose_gt[:3, :3]
        t_gt = pose_gt[:, 3]

        pts_est = self.transform_pts(pts, rot_pred, t_pred)
        pts_gt = self.transform_pts(pts, rot_gt, t_gt)
        error = np.linalg.norm(pts_est - pts_gt, axis=1).mean()
        return error

    def calc_adi(self, pts, pose_pred, pose_gt):
        """
        Average Distance of Model Points for objects with indistinguishable views
        - by Hinterstoisser et al. (ACCV 2012).
        http://www.stefan-hinterstoisser.com/papers/hinterstoisser2012accv.pdf

        :param pts: nx3 ndarray with 3D model points.
        :param pose_pred: Estimated pose (3x3 rot. matrix and 3x1 translation vector).
        :param pose_gt: GT pose (3x3 rot. matrix and 3x1 translation vector).
        :return: Mean average error between the predicted and ground truth pose reduced by symmetry.
        """
        rot_pred = pose_pred[:3, :3]
        t_pred = pose_pred[:, 3]

        rot_gt = pose_gt[:3, :3]
        t_gt = pose_gt[:, 3]

        if pts.shape[0] > 4096:
            indices = np.random.choice(pts.shape[0], 4096, replace=False)
            pts = pts[indices]

        pts_pred = self.transform_pts(pts, rot_pred, t_pred)
        pts_gt = self.transform_pts(pts, rot_gt, t_gt)

        # Calculate distances to the nearest neighbors from pts_gt to pts_est
        nn_index = spatial.cKDTree(pts_pred)
        nn_dists, _ = nn_index.query(pts_gt, k=1)

        error = nn_dists.mean()
        return error

    def calc_rotation_error(self, rot_pred, r_gt):
        """
        Calculate the angular geodesic rotation error between a predicted rotation matrix and the ground truth matrix.
        :paran rot_pred: Predicted rotation matrix (3x3)
        :param rot_gt: Ground truth rotation matrix (3x3)
        """
        assert (rot_pred.shape == r_gt.shape == (3, 3))
        temp = logm(np.dot(np.transpose(rot_pred), r_gt))
        rd_rad = LA.norm(temp, 'fro') / np.sqrt(2)
        rd_deg = rd_rad / np.pi * 180
        return rd_deg

    def evaluate_force_prediction(self, output_path, epoch=None):
        """
        Evaluate force prediction accuracy using Mean Absolute Error (MAE) and Root Mean Square Error (RMSE)
        """
        output_dir = output_path + "/force_evaluation/"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        log_file = open(output_path + "/force_evaluation/force_evaluation.log", 'w')
        json_file = open(output_path + "/force_evaluation/force_evaluation.json", 'w')

        log_file.write('\n* {} *\n {:^}\n* {} *'.format('-' * 100, 'Force Prediction Evaluation', '-' * 100))
        log_file.write("\n")

        forces_pred = copy.deepcopy(self.forces_pred)
        forces_gt = copy.deepcopy(self.forces_gt)
        
        all_force_errors = []
        cls_force_mae = {}
        cls_force_rmse = {}
        cls_force_errors = {}
        evaluation_results = {}

        for cls in self.classes:
            cls_force_errors[cls] = []
            cls_forces_pred = forces_pred[cls]
            cls_forces_gt = forces_gt[cls]
            
            if len(cls_forces_pred) == 0 or len(cls_forces_gt) == 0:
                cls_force_mae[cls] = np.nan
                cls_force_rmse[cls] = np.nan
                log_file.write("Class: {} \t\t No force data available\n".format(cls))
                continue
            
            # Calculate absolute errors for this class
            for force_pred, force_gt in zip(cls_forces_pred, cls_forces_gt):
                abs_error = abs(force_pred - force_gt)
                cls_force_errors[cls].append(abs_error)
                all_force_errors.append(abs_error)
            
            # Calculate MAE and RMSE for this class
            if len(cls_force_errors[cls]) > 0:
                cls_force_mae[cls] = np.mean(cls_force_errors[cls])
                cls_force_rmse[cls] = np.sqrt(np.mean([e**2 for e in cls_force_errors[cls]]))
            else:
                cls_force_mae[cls] = np.nan
                cls_force_rmse[cls] = np.nan
            
            log_file.write("Class: {} \t\t MAE: {:.6f} \t RMSE: {:.6f}\n".format(
                cls, cls_force_mae[cls], cls_force_rmse[cls]))
            
            # Store results for TensorBoard logging
            evaluation_results[cls] = {
                'mae': cls_force_mae[cls],
                'rmse': cls_force_rmse[cls],
                'num_samples': len(cls_force_errors[cls])
            }
            
            # Log to TensorBoard if available
            if self.writer is not None and self.training == True:
                if not self.testing:
                    self.writer.add_scalar(f'ValClassForce/mae/{cls}', cls_force_mae[cls], epoch)
                    self.writer.add_scalar(f'ValClassForce/rmse/{cls}', cls_force_rmse[cls], epoch)
                else:
                    self.writer.add_scalar(f"TestClassForce/mae/{cls}", cls_force_mae[cls])
                    self.writer.add_scalar(f"TestClassForce/rmse/{cls}", cls_force_rmse[cls])

        # Calculate overall metrics
        if len(all_force_errors) > 0:
            total_mae = np.mean(all_force_errors)
            total_rmse = np.sqrt(np.mean([e**2 for e in all_force_errors]))
        else:
            total_mae = np.nan
            total_rmse = np.nan

        log_file.write("Overall:\t\t\t MAE: {:.6f} \t RMSE: {:.6f}\n".format(total_mae, total_rmse))
        
        # Store overall results
        evaluation_results["overall"] = {
            'mae': total_mae,
            'rmse': total_rmse,
            'num_samples': len(all_force_errors)
        }

        log_file.write("\n")
        log_file.close()
        json.dump(evaluation_results, json_file, indent=2)
        json_file.close()
        
        return total_mae, total_rmse

    def calculate_class_avg_force_error(self, output_path, epoch=None):
        """
        Calculate the average force prediction error for each class and overall
        Returns the total average force error
        """
        output_dir = output_path + "/avg_force_error/"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        log_file = open(output_path + "/avg_force_error/avg_force_error.log", 'w')
        json_file = open(output_path + "/avg_force_error/avg_force_error.json", 'w')

        log_file.write('\n* {} *\n {:^}\n* {} *'.format('-' * 100, 'Average Force Prediction Error', '-' * 100))
        log_file.write("\n")

        forces_pred = copy.deepcopy(self.forces_pred)
        forces_gt = copy.deepcopy(self.forces_gt)
        force_errors = []
        cls_force_errors = {}
        avg_force_errors = {}

        for cls in self.classes:
            cls_force_errors[cls] = []
            cls_forces_pred = forces_pred[cls]
            cls_forces_gt = forces_gt[cls]
            
            for force_pred, force_gt in zip(cls_forces_pred, cls_forces_gt):
                abs_error = abs(force_pred - force_gt)
                cls_force_errors[cls].append(abs_error)
                force_errors.append(abs_error)
                
            if len(cls_force_errors[cls]) != 0:
                avg_error = np.mean(cls_force_errors[cls])
                avg_force_errors[cls] = avg_error
            else:
                avg_force_errors[cls] = np.nan
                
            log_file.write("Class: {} \t\t {:.6f}".format(cls, avg_force_errors[cls]))
            log_file.write("\n")
            
            if self.writer is not None and self.training == True:
                if not self.testing:
                    self.writer.add_scalar(f'ValClassForce/avg_force_err/{cls}', avg_force_errors[cls], epoch)
                else:
                    self.writer.add_scalar(f"TestClassForce/avg_force_err/{cls}", avg_force_errors[cls])

        total_avg_error = np.mean(force_errors) if len(force_errors) > 0 else np.nan
        log_file.write("All:\t\t\t\t\t {:.6f}".format(total_avg_error))
        avg_force_errors["mean"] = [total_avg_error]

        log_file.write("\n")
        log_file.close()
        json.dump(avg_force_errors, json_file, indent=2)
        json_file.close()
        return total_avg_error

    def evaluate_force_matrix_prediction(self, output_path, epoch=None):
        """
        Evaluate force matrix prediction with detailed metrics including:
        - Force vector MSE and MAE
        - Force direction accuracy (cosine similarity)  
        - Force existence detection accuracy
        - Force interaction graph accuracy
        """
        output_dir = output_path + "/force_matrix_evaluation/"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        log_file = open(output_path + "/force_matrix_evaluation/force_matrix_evaluation.log", 'w')
        json_file = open(output_path + "/force_matrix_evaluation/force_matrix_evaluation.json", 'w')

        log_file.write('\n* {} *\n {:^}\n* {} *'.format('-' * 100, 'Force Matrix Prediction Evaluation', '-' * 100))
        log_file.write("\n")

        force_matrices_pred = copy.deepcopy(self.force_matrices_pred)
        force_matrices_gt = copy.deepcopy(self.force_matrices_gt)
        
        evaluation_results = {}
        all_vector_errors = []
        all_direction_errors = []
        all_detection_errors = []
        
        force_threshold = 1e-2  # Increased threshold to reduce false positives (was 1e-3)
        
        for cls in self.classes:
            cls_matrices_pred = force_matrices_pred[cls]
            cls_matrices_gt = force_matrices_gt[cls]
            
            if len(cls_matrices_pred) == 0 or len(cls_matrices_gt) == 0:
                log_file.write("Class: {} \t\t No force matrix data available\n".format(cls))
                evaluation_results[cls] = {
                    'vector_mse': np.nan,
                    'vector_mae': np.nan,
                    'direction_accuracy': np.nan,
                    'detection_accuracy': np.nan,
                    'precision': np.nan,
                    'recall': np.nan,
                    'f1_score': np.nan,
                    'num_samples': 0
                }
                continue
            
            cls_vector_errors = []
            cls_direction_accuracies = []
            cls_detection_accuracies = []
            cls_precisions = []
            cls_recalls = []
            cls_f1_scores = []
            
            # Evaluate each force matrix pair
            for pred_matrix, gt_matrix in zip(cls_matrices_pred, cls_matrices_gt):
                # pred_matrix: [N, N, 3], gt_matrix: [N, N, 3]
                
                # 1. Force vector MSE and MAE
                vector_mse = np.mean((pred_matrix - gt_matrix) ** 2)
                vector_mae = np.mean(np.abs(pred_matrix - gt_matrix))
                cls_vector_errors.append({'mse': vector_mse, 'mae': vector_mae})
                all_vector_errors.append({'mse': vector_mse, 'mae': vector_mae})
                
                # 2. Force direction accuracy (only for existing forces)
                gt_norms = np.linalg.norm(gt_matrix, axis=-1)  # [N, N]
                pred_norms = np.linalg.norm(pred_matrix, axis=-1)  # [N, N]
                
                # Mask for positions where forces exist in ground truth
                force_exists_mask = gt_norms > force_threshold
                
                if np.any(force_exists_mask):
                    # Normalize vectors for direction comparison
                    gt_normalized = gt_matrix[force_exists_mask] / (gt_norms[force_exists_mask][..., np.newaxis] + 1e-8)
                    pred_normalized = pred_matrix[force_exists_mask] / (pred_norms[force_exists_mask][..., np.newaxis] + 1e-8)
                    
                    # Compute cosine similarity
                    cosine_similarities = np.sum(gt_normalized * pred_normalized, axis=-1)
                    mean_cosine_sim = np.mean(cosine_similarities)
                    cls_direction_accuracies.append(mean_cosine_sim)
                    all_direction_errors.append(mean_cosine_sim)
                
                # 3. Force existence detection accuracy
                pred_exists_mask = pred_norms > force_threshold
                detection_accuracy = np.mean(force_exists_mask == pred_exists_mask)
                cls_detection_accuracies.append(detection_accuracy)
                all_detection_errors.append(detection_accuracy)
                
                # 4. Precision, Recall, F1 Score for force detection
                true_positives = np.sum(force_exists_mask & pred_exists_mask)
                false_positives = np.sum(~force_exists_mask & pred_exists_mask)
                false_negatives = np.sum(force_exists_mask & ~pred_exists_mask)
                
                precision = true_positives / (true_positives + false_positives + 1e-8)
                recall = true_positives / (true_positives + false_negatives + 1e-8)
                f1_score = 2 * precision * recall / (precision + recall + 1e-8)
                
                cls_precisions.append(precision)
                cls_recalls.append(recall)
                cls_f1_scores.append(f1_score)
            
            # Calculate class-level metrics
            cls_vector_mse = np.mean([e['mse'] for e in cls_vector_errors]) if cls_vector_errors else np.nan
            cls_vector_mae = np.mean([e['mae'] for e in cls_vector_errors]) if cls_vector_errors else np.nan
            cls_direction_acc = np.mean(cls_direction_accuracies) if cls_direction_accuracies else np.nan
            cls_detection_acc = np.mean(cls_detection_accuracies) if cls_detection_accuracies else np.nan
            cls_precision = np.mean(cls_precisions) if cls_precisions else np.nan
            cls_recall = np.mean(cls_recalls) if cls_recalls else np.nan
            cls_f1 = np.mean(cls_f1_scores) if cls_f1_scores else np.nan
            
            # Log class results
            log_file.write("Class: {} \n".format(cls))
            log_file.write("  Vector MSE: {:.6f} \n".format(cls_vector_mse))
            log_file.write("  Vector MAE: {:.6f} \n".format(cls_vector_mae))
            log_file.write("  Direction Accuracy: {:.6f} \n".format(cls_direction_acc))
            log_file.write("  Detection Accuracy: {:.6f} \n".format(cls_detection_acc))
            log_file.write("  Precision: {:.6f} \n".format(cls_precision))
            log_file.write("  Recall: {:.6f} \n".format(cls_recall))
            log_file.write("  F1 Score: {:.6f} \n".format(cls_f1))
            log_file.write("\n")
            
            # Store results
            evaluation_results[cls] = {
                'vector_mse': cls_vector_mse,
                'vector_mae': cls_vector_mae,
                'direction_accuracy': cls_direction_acc,
                'detection_accuracy': cls_detection_acc,
                'precision': cls_precision,
                'recall': cls_recall,
                'f1_score': cls_f1,
                'num_samples': len(cls_vector_errors)
            }
            
            # Log to TensorBoard if available
            if self.writer is not None and self.training == True:
                if not self.testing:
                    self.writer.add_scalar(f'ValClassForceMatrix/vector_mse/{cls}', cls_vector_mse, epoch)
                    self.writer.add_scalar(f'ValClassForceMatrix/vector_mae/{cls}', cls_vector_mae, epoch)
                    self.writer.add_scalar(f'ValClassForceMatrix/direction_acc/{cls}', cls_direction_acc, epoch)
                    self.writer.add_scalar(f'ValClassForceMatrix/detection_acc/{cls}', cls_detection_acc, epoch)
                    self.writer.add_scalar(f'ValClassForceMatrix/precision/{cls}', cls_precision, epoch)
                    self.writer.add_scalar(f'ValClassForceMatrix/recall/{cls}', cls_recall, epoch)
                    self.writer.add_scalar(f'ValClassForceMatrix/f1_score/{cls}', cls_f1, epoch)
                else:
                    self.writer.add_scalar(f"TestClassForceMatrix/vector_mse/{cls}", cls_vector_mse)
                    self.writer.add_scalar(f"TestClassForceMatrix/vector_mae/{cls}", cls_vector_mae)
                    self.writer.add_scalar(f"TestClassForceMatrix/direction_acc/{cls}", cls_direction_acc)
                    self.writer.add_scalar(f"TestClassForceMatrix/detection_acc/{cls}", cls_detection_acc)
                    self.writer.add_scalar(f"TestClassForceMatrix/precision/{cls}", cls_precision)
                    self.writer.add_scalar(f"TestClassForceMatrix/recall/{cls}", cls_recall)
                    self.writer.add_scalar(f"TestClassForceMatrix/f1_score/{cls}", cls_f1)

        # Calculate overall metrics
        overall_vector_mse = np.mean([e['mse'] for e in all_vector_errors]) if all_vector_errors else np.nan
        overall_vector_mae = np.mean([e['mae'] for e in all_vector_errors]) if all_vector_errors else np.nan
        overall_direction_acc = np.mean(all_direction_errors) if all_direction_errors else np.nan
        overall_detection_acc = np.mean(all_detection_errors) if all_detection_errors else np.nan
        
        # Calculate overall precision, recall, f1 from class-level results
        valid_classes = [cls for cls in self.classes if not np.isnan(evaluation_results[cls]['precision'])]
        overall_precision = np.mean([evaluation_results[cls]['precision'] for cls in valid_classes]) if valid_classes else np.nan
        overall_recall = np.mean([evaluation_results[cls]['recall'] for cls in valid_classes]) if valid_classes else np.nan
        overall_f1 = np.mean([evaluation_results[cls]['f1_score'] for cls in valid_classes]) if valid_classes else np.nan

        log_file.write("Overall Results:\n")
        log_file.write("  Vector MSE: {:.6f} \n".format(overall_vector_mse))
        log_file.write("  Vector MAE: {:.6f} \n".format(overall_vector_mae))
        log_file.write("  Direction Accuracy: {:.6f} \n".format(overall_direction_acc))
        log_file.write("  Detection Accuracy: {:.6f} \n".format(overall_detection_acc))
        log_file.write("  Precision: {:.6f} \n".format(overall_precision))
        log_file.write("  Recall: {:.6f} \n".format(overall_recall))
        log_file.write("  F1 Score: {:.6f} \n".format(overall_f1))
        
        # Store overall results
        evaluation_results["overall"] = {
            'vector_mse': overall_vector_mse,
            'vector_mae': overall_vector_mae,
            'direction_accuracy': overall_direction_acc,
            'detection_accuracy': overall_detection_acc,
            'precision': overall_precision,
            'recall': overall_recall,
            'f1_score': overall_f1,
            'num_samples': len(all_vector_errors)
        }

        log_file.write("\n")
        log_file.close()
        json.dump(evaluation_results, json_file, indent=2)
        json_file.close()
        
        return evaluation_results

    def calculate_class_avg_force_matrix_error(self, output_path, epoch=None):
        """
        Calculate simplified average force matrix error for quick monitoring
        """
        output_dir = output_path + "/avg_force_matrix_error/"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        log_file = open(output_path + "/avg_force_matrix_error/avg_force_matrix_error.log", 'w')
        json_file = open(output_path + "/avg_force_matrix_error/avg_force_matrix_error.json", 'w')

        log_file.write('\n* {} *\n {:^}\n* {} *'.format('-' * 100, 'Average Force Matrix Error', '-' * 100))
        log_file.write("\n")

        force_matrices_pred = copy.deepcopy(self.force_matrices_pred)
        force_matrices_gt = copy.deepcopy(self.force_matrices_gt)
        force_matrix_errors = []
        cls_force_matrix_errors = {}
        avg_force_matrix_errors = {}

        for cls in self.classes:
            cls_force_matrix_errors[cls] = []
            cls_matrices_pred = force_matrices_pred[cls]
            cls_matrices_gt = force_matrices_gt[cls]
            
            for pred_matrix, gt_matrix in zip(cls_matrices_pred, cls_matrices_gt):
                # Calculate mean absolute error for this matrix pair
                mae = np.mean(np.abs(pred_matrix - gt_matrix))
                cls_force_matrix_errors[cls].append(mae)
                force_matrix_errors.append(mae)
                
            if len(cls_force_matrix_errors[cls]) != 0:
                avg_error = np.mean(cls_force_matrix_errors[cls])
                avg_force_matrix_errors[cls] = avg_error
            else:
                avg_force_matrix_errors[cls] = np.nan
                
            log_file.write("Class: {} \t\t {:.6f}".format(cls, avg_force_matrix_errors[cls]))
            log_file.write("\n")
            
            if self.writer is not None and self.training == True:
                if not self.testing:
                    self.writer.add_scalar(f'ValClassForceMatrix/avg_matrix_err/{cls}', avg_force_matrix_errors[cls], epoch)
                else:
                    self.writer.add_scalar(f"TestClassForceMatrix/avg_matrix_err/{cls}", avg_force_matrix_errors[cls])

        total_avg_error = np.mean(force_matrix_errors) if len(force_matrix_errors) > 0 else np.nan
        log_file.write("All:\t\t\t\t\t {:.6f}".format(total_avg_error))
        avg_force_matrix_errors["mean"] = [total_avg_error]

        log_file.write("\n")
        log_file.close()
        json.dump(avg_force_matrix_errors, json_file, indent=2)
        json_file.close()
        return total_avg_error





