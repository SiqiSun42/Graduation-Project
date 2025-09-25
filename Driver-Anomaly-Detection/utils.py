import csv
import numpy as np
import os
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import auc

def l2_normalize(x, dim=1):
    return x / torch.sqrt(torch.sum(x**2, dim=dim).unsqueeze(dim))

def adjust_learning_rate(optimizer, lr_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_rate


class Logger(object):
    """Logger object for training process, supporting resume training"""
    def __init__(self, path, header, resume=False):
        """
        :param path: logging file path
        :param header: a list of tags for values to track
        :param resume: a flag controling whether to create a new
        file or continue recording after the latest step
        """
        self.log_file = None
        self.resume = resume
        self.header = header
        if not self.resume:
            self.log_file = open(path, 'w')
            self.logger = csv.writer(self.log_file, delimiter='\t')
            self.logger.writerow(self.header)
        else:
            self.log_file = open(path, 'a+')
            self.log_file.seek(0, os.SEEK_SET)
            reader = csv.reader(self.log_file, delimiter='\t')
            self.header = next(reader)
            # move back to the end of file
            self.log_file.seek(0, os.SEEK_END)
            self.logger = csv.writer(self.log_file, delimiter='\t')

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for tag in self.header:
            assert tag in values, 'Please give the right value as defined'
            write_values.append(values[tag])
        self.logger.writerow(write_values)
        self.log_file.flush()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def _construct_depth_model(base_model):
    # modify the first convolution kernels for Depth input
    modules = list(base_model.modules())
    first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv3d),
                                 list(range(len(modules)))))[0]
    conv_layer = modules[first_conv_idx]
    container = modules[first_conv_idx - 1]
    # modify parameters, assume the first blob contains the convolution kernels
    motion_length = 1
    params = [x.clone() for x in conv_layer.parameters()]
    kernel_size = params[0].size()
    new_kernel_size = kernel_size[:1] + (1*motion_length,  ) + kernel_size[2:]
    new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
    new_conv = nn.Conv3d(1, conv_layer.out_channels, conv_layer.kernel_size, conv_layer.stride,
                         conv_layer.padding, bias=True if len(params) == 2 else False)
    new_conv.weight.data = new_kernels
    if len(params) == 2:
        new_conv.bias.data = params[1].data # add bias if neccessary
    layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name
    # replace the first convlution layer
    setattr(container, layer_name, new_conv)
    return base_model

def get_fusion_label(csv_path):
    """
    Read the csv file and return labels
    :param csv_path: path of csv file
    :return: ground truth labels
    """
    gt = np.zeros(360000)
    base = -10000
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[-1] == '':
                continue
            if row[1] != '':
                base += 10000
            if row[4] == 'N':
                gt[base + int(row[2]):base + int(row[3]) + 1] = 1
            else:
                continue
    return gt

# def evaluate(score, label, whether_plot):
#     """
#     Compute Accuracy as well as AUC by evaluating the scores
#     :param score: scores of each frame in videos which are computed as the cosine similarity between encoded test vector and mean vector of normal driving
#     :param label: ground truth
#     :param whether_plot: whether plot the AUC curve
#     :return: best accuracy, corresponding threshold, AUC
#     """
#     thresholds = np.arange(0., 1., 0.01)
#     best_acc = 0.
#     best_threshold = 0.
#     for threshold in thresholds:
#         prediction = score >= threshold
#         #print(prediction)
#         correct = prediction == label
#
#         acc = (np.sum(correct) / correct.shape[0] * 100)
#         if acc > best_acc:
#             best_acc = acc
#             best_threshold = threshold
#
#     fpr, tpr, thresholds = metrics.roc_curve(label, score, pos_label=1)
#     AUC = auc(fpr, tpr)
#
#     if whether_plot:
#         plt.plot(fpr, tpr, color='r')
#         plt.fill_between(fpr, tpr, color='r', y2=0, alpha=0.3)
#         plt.plot(np.array([0., 1.]), np.array([0., 1.]), color='b', linestyle='dashed')
#         plt.tick_params(labelsize=23)
#         plt.text(0.9, 0.1, f'AUC: {round(AUC, 4)}', fontsize=25)
#         plt.xlabel('False Positive Rate', fontsize=25)
#         plt.ylabel('True Positive Rate', fontsize=25)
#         plt.show()
#     return best_acc, best_threshold, AUC

# def evaluate(score, label, whether_plot):
#     """
#     Compute Accuracy, Normal (negative class) Accuracy, and Anormal (positive class) Accuracy as well as AUC by evaluating the scores
#     :param score: scores of each frame in videos which are computed as the cosine similarity between encoded test vector and mean vector of normal driving
#     :param label: ground truth
#     :param whether_plot: whether plot the AUC curve
#     :return: best accuracy, corresponding threshold, AUC, accuracy for normal, accuracy for anormal
#     """
#     thresholds = np.arange(0., 1., 0.01)
#     best_acc = 0.
#     best_threshold = 0.
#     best_normal_acc = 0.
#     best_anormal_acc = 0.
#
#     for threshold in thresholds:
#         prediction = score >= threshold
#         correct = prediction == label
#         normal_correct = (prediction == label) & (label == 1)
#         anormal_correct = (prediction == label) & (label == 0)
#
#         acc = np.mean(correct) * 100
#         normal_acc = np.sum(normal_correct) / np.sum(label == 1) * 100  if np.sum(label == 1) > 0 else 0
#         anormal_acc = np.sum(anormal_correct) / np.sum(label == 0) * 100 if np.sum(label == 0) > 0 else 0
#
#         if acc > best_acc:
#             best_acc = acc
#             best_threshold = threshold
#             best_normal_acc = normal_acc
#             best_anormal_acc = anormal_acc
#
#     fpr, tpr, thresholds = metrics.roc_curve(label, score, pos_label=1)
#     AUC = metrics.auc(fpr, tpr)
#
#     if whether_plot:
#         plt.plot(fpr, tpr, color='red', label='ROC Curve')
#         plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
#         plt.xlabel('False Positive Rate')
#         plt.ylabel('True Positive Rate')
#         plt.show()
#
#     return best_acc, best_threshold, AUC, best_normal_acc, best_anormal_acc

# def evaluate(score, label, whether_plot):
#     """
#     Compute Accuracy, Normal (negative class) Accuracy, Anormal (positive class) Accuracy, and AUC by evaluating the scores.
#     :param score: Scores of each frame in videos which are computed as the cosine similarity between encoded test vector and mean vector of normal driving.
#     :param label: Ground truth (0 for normal, 1 for anormal).
#     :param whether_plot: Whether to plot the AUC curve.
#     :return: Best accuracy, corresponding threshold, AUC, accuracy for normal, accuracy for anormal.
#     """
#     thresholds = np.arange(0., 1., 0.01)
#     total_correct_n = np.zeros(thresholds.shape[0])  # 正常样本正确预测数
#     total_correct_a = np.zeros(thresholds.shape[0])  # 异常样本正确预测数
#     total_pred_n = np.zeros(thresholds.shape[0])  # 预测为正常的总数
#     total_pred_a = np.zeros(thresholds.shape[0])  # 预测为异常的总数
#     total_n = np.sum(label == 1)  # 总正常样本数，根据上面的label定义，正常样本为1
#     total_a = np.sum(label == 0)  # 总异常样本数，根据上面的label定义，异常样本为0
#
#     for i, threshold in enumerate(thresholds):
#         # Make predictions
#         prediction = score >= threshold
#         correct = prediction == label
#
#         # Count correct predictions
#         total_correct_n[i] = np.sum(correct[label == 1])  # 正常样本正确预测
#         total_correct_a[i] = np.sum(correct[label == 0])  # 异常样本正确预测
#
#         # Count predictions
#         total_pred_n[i] = np.sum(prediction == 1)  # 预测为正常
#         total_pred_a[i] = np.sum(prediction == 0)  # 预测为异常
#
#     # Compute accuracy for each threshold
#     acc_n = total_correct_n / total_n if total_n > 0 else np.zeros(thresholds.shape[0])
#     acc_a = total_correct_a / total_a if total_a > 0 else np.zeros(thresholds.shape[0])
#     acc = (total_correct_n + total_correct_a) / (total_n + total_a)
#
#     # Find best accuracy and corresponding threshold
#     best_acc = np.max(acc) * 100
#     idx = np.argmax(acc)
#     best_threshold = thresholds[idx]
#
#     best_normal_acc = acc_n[idx] * 100
#     best_anormal_acc = acc_a[idx] * 100
#
#     # Compute AUC using ROC curve
#     fpr, tpr, _ = metrics.roc_curve(label, score, pos_label=1)
#     AUC = metrics.auc(fpr, tpr)
#
#     # Plot ROC curve if required
#     if whether_plot:
#         plt.plot(fpr, tpr, color='red', label='ROC Curve')
#         plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
#         plt.xlabel('False Positive Rate')
#         plt.ylabel('True Positive Rate')
#         plt.title('ROC Curve')
#         plt.legend()
#         plt.grid()
#         plt.show()
#
#     return best_acc, best_threshold, AUC, best_normal_acc, best_anormal_acc

def evaluate(score, label, whether_plot):
    """
    Compute Accuracy, Normal (negative class) Accuracy, Anormal (positive class) Accuracy, AUC, and precision metrics by evaluating the scores.
    :param score: Scores of each frame in videos which are computed as the cosine similarity between encoded test vector and mean vector of normal driving.
    :param label: Ground truth (0 for normal, 1 for anormal).
    :param whether_plot: Whether to plot the AUC curve.
    :return: Best accuracy, corresponding threshold, AUC, accuracy for normal, accuracy for anormal, precision for normal, precision for anormal.
    """
    thresholds = np.arange(0., 1., 0.01)
    total_correct_n = np.zeros(thresholds.shape[0])  # 正常样本正确预测数
    total_correct_a = np.zeros(thresholds.shape[0])  # 异常样本正确预测数
    total_pred_n = np.zeros(thresholds.shape[0])  # 预测为正常的总数
    total_pred_a = np.zeros(thresholds.shape[0])  # 预测为异常的总数
    total_n = np.sum(label == 1)  # 总正常样本数，根据上面的label定义，正常样本为1
    total_a = np.sum(label == 0)  # 总异常样本数，根据上面的label定义，异常样本为0

    # Initialize metrics for recall and miss rate
    normal_recall = []
    normal_miss_rate = []
    anormal_recall = []
    anormal_miss_rate = []
    normal_precision = []
    anormal_precision = []

    # 计算“正常”，“模糊异常”，“典型异常”的数量和比例
    normal_count = []  # 正常样本计数
    fuzzy_anomalous_count = []  # 模糊异常样本计数
    typical_anomalous_count = []  # 典型异常样本计数
    normal_ratio = []  # 正常样本比例
    fuzzy_anomalous_ratio = []  # 模糊异常样本比例
    typical_anomalous_ratio = []  # 典型异常样本比例
    threshold_offset = 0.2  # 偏差值，可修改

    # # Calculate score distribution in bins (0.0 to 1.0 with 0.1 intervals)
    # score_bins = np.arange(0, 1.1, 0.1)
    #
    # # Calculate and output score distribution
    # score_distribution = []
    # ratio = []  # To store the ratio for each bin
    # for i in range(len(score_bins) - 1):
    #     lower_bound = score_bins[i]
    #     upper_bound = score_bins[i + 1]
    #     count = np.sum((score >= lower_bound) & (score < upper_bound))
    #     ratio.append(count / len(score))  # Calculate the ratio for each bin
    #     score_distribution.append((lower_bound, upper_bound, count))  # Store bin info (range, count)

    for i, threshold in enumerate(thresholds):
        # Make predictions
        prediction = score >= threshold
        correct = prediction == label

        # Count correct predictions
        total_correct_n[i] = np.sum(correct[label == 1])  # 正常样本正确预测
        total_correct_a[i] = np.sum(correct[label == 0])  # 异常样本正确预测

        # Count predictions
        total_pred_n[i] = np.sum(prediction == 1)  # 预测为正常
        total_pred_a[i] = np.sum(prediction == 0)  # 预测为异常

        # Calculate recall and miss rate for normal samples
        recall_n = total_correct_n[i] / total_n if total_n > 0 else 0
        miss_rate_n = 1 - recall_n
        normal_recall.append(recall_n)
        normal_miss_rate.append(miss_rate_n)

        # Calculate recall and miss rate for anormal samples
        recall_a = total_correct_a[i] / total_a if total_a > 0 else 0
        miss_rate_a = 1 - recall_a
        anormal_recall.append(recall_a)
        anormal_miss_rate.append(miss_rate_a)

        # Calculate precision for normal and anormal samples
        precision_n = total_correct_n[i] / total_pred_n[i] if total_pred_n[i] > 0 else 0
        precision_a = total_correct_a[i] / total_pred_a[i] if total_pred_a[i] > 0 else 0
        normal_precision.append(precision_n)
        anormal_precision.append(precision_a)

        # 统计每个类别的样本数量和比例
        normal_samples = np.sum((score >= threshold))  # 高于阈值的样本为正常
        fuzzy_anomalous_samples = np.sum((score < threshold) & (score >= threshold - threshold_offset))  # 模糊异常
        typical_anomalous_samples = np.sum(score < threshold - threshold_offset)  # 典型异常

        # 计算各自比例
        normal_count.append(normal_samples)
        fuzzy_anomalous_count.append(fuzzy_anomalous_samples)
        typical_anomalous_count.append(typical_anomalous_samples)

        normal_ratio.append(normal_samples / len(score))
        fuzzy_anomalous_ratio.append(fuzzy_anomalous_samples / len(score))
        typical_anomalous_ratio.append(typical_anomalous_samples / len(score))

    # Compute accuracy for each threshold
    acc_n = total_correct_n / total_n if total_n > 0 else np.zeros(thresholds.shape[0])
    acc_a = total_correct_a / total_a if total_a > 0 else np.zeros(thresholds.shape[0])
    acc = (total_correct_n + total_correct_a) / (total_n + total_a)

    # 设置权重，重点关注异常样本的准确率，但也兼顾正常样本和总体准确率
    weight_normal = 0.5  # 正常样本的权重
    weight_anomalous = 0.5  # 异常样本的权重
    weight_overall = 0  # 总体准确率的权重

    # 计算加权后的准确率
    weighted_acc = (weight_normal * acc_n) + (weight_anomalous * acc_a) + (weight_overall * acc)

    # 找到最佳加权准确率和对应的阈值
    #best_acc_weighted = np.max(weighted_acc) * 100  # 最佳加权准确率
    idx = np.argmax(weighted_acc)  # 对应的阈值索引
    best_acc = acc[idx] * 100  # 最佳准确率
    best_threshold = thresholds[idx]  # 最佳阈值

    # # Find best accuracy and corresponding threshold
    # best_acc = np.max(acc) * 100
    # idx = np.argmax(acc)
    # best_threshold = thresholds[idx]

    # # Find best accuracy for anomalous samples (acc_a) and corresponding threshold
    # best_acc = np.max(acc_a) * 100  # Best accuracy for anomalous samples
    # idx = np.argmax(acc_a)  # Index corresponding to best anomalous accuracy
    # best_threshold = thresholds[idx]  # Best threshold for anomalous samples

    best_normal_acc = acc_n[idx] * 100
    best_anormal_acc = acc_a[idx] * 100
    best_normal_precision = normal_precision[idx]
    best_anormal_precision = anormal_precision[idx]
    best_normal_ratio = normal_ratio[idx]
    best_fuzzy_anomalous_ratio = fuzzy_anomalous_ratio[idx]
    best_typical_anomalous_ratio = typical_anomalous_ratio[idx]

    # Compute AUC using ROC curve
    fpr, tpr, roc_thresholds = metrics.roc_curve(label, score, pos_label=1)
    AUC = metrics.auc(fpr, tpr)

    # Plot ROC curve if required
    if whether_plot:
        # 找到最接近 best_threshold 的 roc_thresholds 索引
        roc_idx = (np.abs(roc_thresholds - best_threshold)).argmin()

        # 获取 FPR 和 TPR
        best_fpr = fpr[roc_idx]
        best_tpr = tpr[roc_idx]

        # Plot ROC Curve
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='red', label='ROC Curve')
        plt.plot([0, 1], [0, 1], color='blue', linestyle='--', label='Random Guess')
        plt.scatter(best_fpr, best_tpr, color='green',
                    label=f'Best Threshold: {best_threshold:.2f} (FPR={best_fpr:.2f}, TPR={best_tpr:.2f})')
        plt.axvline(x=best_fpr, color='green', linestyle=':')
        plt.axhline(y=best_tpr, color='green', linestyle=':')
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title(f'ROC Curve (Best Threshold: {best_threshold:.2f})')
        plt.legend()
        plt.grid()
        plt.show()

        # Plot recall and miss rate for normal samples
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, normal_recall, label='Normal Recall', color='green')
        plt.plot(thresholds, normal_miss_rate, label='Normal Miss Rate', color='orange')
        plt.scatter(best_threshold, normal_recall[idx], color='green',
                    label=f'normal_recall: {normal_recall[idx]:.2f}')
        plt.scatter(best_threshold, normal_miss_rate[idx], color='orange',
                    label=f'normal_miss_rate: {normal_miss_rate[idx]:.2f}')
        plt.axvline(x=best_threshold, color='red', linestyle='--', label=f'Best Threshold: {best_threshold:.2f}')
        plt.xlabel('Threshold')
        plt.ylabel('Rate')
        plt.title('Normal Samples: Recall and Miss Rate')
        plt.legend()
        plt.grid()
        plt.show()

        # Plot recall and miss rate for anormal samples
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, anormal_recall, label='Anormal Recall', color='purple')
        plt.plot(thresholds, anormal_miss_rate, label='Anormal Miss Rate', color='brown')
        plt.scatter(best_threshold, anormal_recall[idx], color='purple',
                    label=f'anormal_recall: {anormal_recall[idx]:.2f}')
        plt.scatter(best_threshold, anormal_miss_rate[idx], color='brown',
                    label=f'anormal_miss_rate: {anormal_miss_rate[idx]:.2f}')
        plt.axvline(x=best_threshold, color='red', linestyle='--', label=f'Best Threshold: {best_threshold:.2f}')
        plt.xlabel('Threshold')
        plt.ylabel('Rate')
        plt.title('Anormal Samples: Recall and Miss Rate')
        plt.legend()
        plt.grid()
        plt.show()

    return best_acc, best_threshold, AUC, best_normal_acc, best_anormal_acc, best_normal_precision, best_anormal_precision, best_normal_ratio, best_fuzzy_anomalous_ratio, best_typical_anomalous_ratio



def post_process(score, window_size=6):
    """
    post process the score
    :param score: scores of each frame in videos
    :param window_size: window size
    :param momentum: momentum factor
    :return: post processed score
    """
    processed_score = np.zeros(score.shape)
    for i in range(0, len(score)):
        processed_score[i] = np.mean(score[max(0, i-window_size+1):i+1])

    return processed_score


def get_score(score_folder, mode):
    """
    !!!Be used only when scores exist!!!
    Get the corresponding scores according to requiements
    :param score_folder: the folder where the scores are saved
    :param mode: top_d | top_ir | front_d | front_ir | fusion_top | fusion_front | fusion_d | fusion_ir | fusion_all
    :return: the corresponding scores according to requirements
    """
    if mode not in ['top_d', 'top_ir', 'front_d', 'front_ir', 'fusion_top', 'fusion_front', 'fusion_d', 'fusion_ir', 'fusion_all']:
        print('Please enter correct mode: top_d | top_ir | front_d | front_ir | fusion_top | fusion_front | fusion_d | fusion_ir | fusion_all')
        return
    if mode == 'top_d':
        score = np.load(os.path.join(score_folder + '/score_top_d.npy'))
    elif mode == 'top_ir':
        score = np.load(os.path.join(score_folder + '/score_top_IR.npy'))
    elif mode == 'front_d':
        score = np.load(os.path.join(score_folder + '/score_front_d.npy'))
    elif mode == 'front_ir':
        score = np.load(os.path.join(score_folder + '/score_front_IR.npy'))
    elif mode == 'fusion_top':
        score1 = np.load(os.path.join(score_folder + '/score_top_d.npy'))
        score2 = np.load(os.path.join(score_folder + '/score_top_IR.npy'))
        score = np.mean((score1, score2), axis = 0)
    elif mode == 'fusion_front':
        score3 = np.load(os.path.join(score_folder + '/score_front_d.npy'))
        score4 = np.load(os.path.join(score_folder + '/score_front_IR.npy'))
        score = np.mean((score3, score4), axis=0)
    elif mode == 'fusion_d':
        score1 = np.load(os.path.join(score_folder + '/score_top_d.npy'))
        score3 = np.load(os.path.join(score_folder + '/score_front_d.npy'))
        score = np.mean((score1, score3), axis=0)
    elif mode == 'fusion_ir':
        score2 = np.load(os.path.join(score_folder + '/score_top_IR.npy'))
        score4 = np.load(os.path.join(score_folder + '/score_front_IR.npy'))
        score = np.mean((score2, score4), axis=0)
    elif mode == 'fusion_all':
        score1 = np.load(os.path.join(score_folder + '/score_top_d.npy'))
        score2 = np.load(os.path.join(score_folder + '/score_top_IR.npy'))
        score3 = np.load(os.path.join(score_folder + '/score_front_d.npy'))
        score4 = np.load(os.path.join(score_folder + '/score_front_IR.npy'))
        score = np.mean((score1, score2, score3, score4), axis=0)

    return score


def get_fusion_score(score_folder, mode):
    """
    Get the corresponding fused scores with a focus on anomaly detection.
    This function uses the minimum value across modalities for the fusion strategy.
    :param score_folder: the folder where the scores are saved
    :param mode: top_d | top_ir | front_d | front_ir | fusion_top | fusion_front | fusion_d | fusion_ir | fusion_all
    :return: the corresponding fused scores
    """
    if mode not in ['top_d', 'top_ir', 'front_d', 'front_ir', 'fusion_top', 'fusion_front', 'fusion_d', 'fusion_ir',
                    'fusion_all']:
        print(
            'Please enter correct mode: top_d | top_ir | front_d | front_ir | fusion_top | fusion_front | fusion_d | fusion_ir | fusion_all')
        return

    if mode == 'top_d':
        score = np.load(os.path.join(score_folder, 'score_top_d.npy'))
    elif mode == 'top_ir':
        score = np.load(os.path.join(score_folder, 'score_top_IR.npy'))
    elif mode == 'front_d':
        score = np.load(os.path.join(score_folder, 'score_front_d.npy'))
    elif mode == 'front_ir':
        score = np.load(os.path.join(score_folder, 'score_front_IR.npy'))
    elif mode == 'fusion_top':
        score1 = np.load(os.path.join(score_folder, 'score_top_d.npy'))
        score2 = np.load(os.path.join(score_folder, 'score_top_IR.npy'))
        score = np.min((score1, score2), axis=0)  # Fusion strategy: use the minimum score
    elif mode == 'fusion_front':
        score3 = np.load(os.path.join(score_folder, 'score_front_d.npy'))
        score4 = np.load(os.path.join(score_folder, 'score_front_IR.npy'))
        score = np.min((score3, score4), axis=0)  # Fusion strategy: use the minimum score
    elif mode == 'fusion_d':
        score1 = np.load(os.path.join(score_folder, 'score_top_d.npy'))
        score3 = np.load(os.path.join(score_folder, 'score_front_d.npy'))
        score = np.min((score1, score3), axis=0)  # Fusion strategy: use the minimum score
    elif mode == 'fusion_ir':
        score2 = np.load(os.path.join(score_folder, 'score_top_IR.npy'))
        score4 = np.load(os.path.join(score_folder, 'score_front_IR.npy'))
        score = np.min((score2, score4), axis=0)  # Fusion strategy: use the minimum score
    elif mode == 'fusion_all':
        score1 = np.load(os.path.join(score_folder, 'score_top_d.npy'))
        score2 = np.load(os.path.join(score_folder, 'score_top_IR.npy'))
        score3 = np.load(os.path.join(score_folder, 'score_front_d.npy'))
        score4 = np.load(os.path.join(score_folder, 'score_front_IR.npy'))
        score = np.min((score1, score2, score3, score4), axis=0)  # Fusion strategy: use the minimum score

    return score

def get_fusion_score_weighted(score_folder, mode, weights):
    """
    Get the corresponding fused scores with a focus on anomaly detection using weighted fusion.
    :param score_folder: the folder where the scores are saved
    :param mode: top_d | top_ir | front_d | front_ir | fusion_top | fusion_front | fusion_d | fusion_ir | fusion_all
    :param weights: list or array of weights for each modality (length must be 4)
    :return: the corresponding fused scores
    """

    if mode not in ['top_d', 'top_ir', 'front_d', 'front_ir', 'fusion_top', 'fusion_front', 'fusion_d', 'fusion_ir',
                    'fusion_all']:
        print(
            'Please enter correct mode: top_d | top_ir | front_d | front_ir | fusion_top | fusion_front | fusion_d | fusion_ir | fusion_all')
        return

    # Check if weights length is correct
    if len(weights) != 4:
        print("The number of weights must be 4, corresponding to each modality.")
        return

    if mode == 'top_d':
        score = np.load(os.path.join(score_folder, 'score_top_d.npy'))
    elif mode == 'top_ir':
        score = np.load(os.path.join(score_folder, 'score_top_IR.npy'))
    elif mode == 'front_d':
        score = np.load(os.path.join(score_folder, 'score_front_d.npy'))
    elif mode == 'front_ir':
        score = np.load(os.path.join(score_folder, 'score_front_IR.npy'))
    elif mode == 'fusion_top':
        score1 = np.load(os.path.join(score_folder, 'score_top_d.npy'))
        score2 = np.load(os.path.join(score_folder, 'score_top_IR.npy'))

        # Compute weighted fusion for top_d and top_IR using weights[0] and weights[1]
        weighted_score = (weights[0] * score1 + weights[1] * score2)
        score = weighted_score / (weights[0] + weights[1])  # Normalize by the sum of weights

    elif mode == 'fusion_front':
        score3 = np.load(os.path.join(score_folder, 'score_front_d.npy'))
        score4 = np.load(os.path.join(score_folder, 'score_front_IR.npy'))

        # Compute weighted fusion for front_d and front_IR using weights[2] and weights[3]
        weighted_score = (weights[2] * score3 + weights[3] * score4)
        score = weighted_score / (weights[2] + weights[3])  # Normalize by the sum of weights

    elif mode == 'fusion_d':
        score1 = np.load(os.path.join(score_folder, 'score_top_d.npy'))
        score3 = np.load(os.path.join(score_folder, 'score_front_d.npy'))

        # Compute weighted fusion for top_d and front_d using weights[0] and weights[2]
        weighted_score = (weights[0] * score1 + weights[2] * score3)
        score = weighted_score / (weights[0] + weights[2])  # Normalize by the sum of weights

    elif mode == 'fusion_ir':
        score2 = np.load(os.path.join(score_folder, 'score_top_IR.npy'))
        score4 = np.load(os.path.join(score_folder, 'score_front_IR.npy'))

        # Compute weighted fusion for top_IR and front_IR using weights[1] and weights[3]
        weighted_score = (weights[1] * score2 + weights[3] * score4)
        score = weighted_score / (weights[1] + weights[3])  # Normalize by the sum of weights

    elif mode == 'fusion_all':
        score1 = np.load(os.path.join(score_folder, 'score_top_d.npy'))
        score2 = np.load(os.path.join(score_folder, 'score_top_IR.npy'))
        score3 = np.load(os.path.join(score_folder, 'score_front_d.npy'))
        score4 = np.load(os.path.join(score_folder, 'score_front_IR.npy'))

        # Compute weighted fusion for all four modalities using weights[0], weights[1], weights[2], and weights[3]
        weighted_score = (weights[0] * score1 + weights[1] * score2 + weights[2] * score3 + weights[3] * score4)
        score = weighted_score / (weights[0] + weights[1] + weights[2] + weights[3])  # Normalize by the sum of weights

    return score








