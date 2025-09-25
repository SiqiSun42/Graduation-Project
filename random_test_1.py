from utils import  get_fusion_label, post_process, evaluate,get_score,get_fusion_score, get_fusion_score_weighted

gt = get_fusion_label('/root/autodl-tmp/dataset/DAD/LABEL.csv')

score_folder = './score/'
hashmap = {'top_d': 'Top(D)',
            'top_ir': 'Top(IR)',
            'fusion_top': 'Top(DIR)',
            'front_d': 'Front(D)',
            'front_ir': 'Front(IR)',
            'fusion_front': 'Front(DIR)',
            'fusion_d': 'Fusion(D)',
            'fusion_ir': 'Fusion(IR)',
            'fusion_all': 'Fusion(DIR)'
            }

def optimize_weights(score_folder, hashmap, gt, max_iterations, weight_increment):
    # 初始权重设置
    weights = [0.25, 0.25, 0.25, 0.25]  # 对应 top_d, top_ir, front_d, front_ir 的权重

    # 用于记录最好的结果
    best_acc_history = []
    best_result = None
    best_acc = 0

    # 循环迭代
    for iteration in range(max_iterations):
        #print(f'interation: {iteration + 1}')
        # 调用evaluate_scores进行评估
        results = evaluate_scores(score_folder, hashmap, gt, weights)

        # 计算Fusion(DIR)的结果
        fusion_best_acc = results['Fusion(DIR)']['best_acc']

        # 保存当前结果
        best_acc_history.append(fusion_best_acc)

        #比较得出最好结果
        if fusion_best_acc > best_acc or best_result is None:
            best_result = results
            best_acc = fusion_best_acc
            #best_weights = weights

        # 如果当前的Fusion(DIR)表现较差，停止循环
        if len(best_acc_history) > 2 and fusion_best_acc < best_acc_history[-1] and fusion_best_acc < best_acc_history[-2]:
            print(f"Stopped at iteration {iteration}")
            print(f"Best Acc after optimization: {best_acc}")
            return best_result

        # 调整权重
        # 比较每一对基础视角的 Best Acc，更新权重
        if results['Top(D)']['best_acc'] > results['Top(IR)']['best_acc']:
            weights[0] += weight_increment  # Top(D) 权重增加
            weights[1] -= weight_increment  # Top(IR) 权重减少
        else:
            weights[0] -= weight_increment  # Top(D) 权重减少
            weights[1] += weight_increment  # Top(IR) 权重增加

        if results['Front(D)']['best_acc'] > results['Front(IR)']['best_acc']:
            weights[2] += weight_increment  # Front(D) 权重增加
            weights[3] -= weight_increment  # Front(IR) 权重减少
        else:
            weights[2] -= weight_increment  # Front(D) 权重减少
            weights[3] += weight_increment  # Front(IR) 权重增加

        #print(f'weights: {weights}')  # 打印当前权重

        # # 归一化权重，使其总和为1
        # total_weight = sum(weights)
        # weights = [w / total_weight for w in weights]

        print(f"Iteration {iteration + 1} complete, Fusion(DIR) Best Acc: {fusion_best_acc}")

    # 输出最终的最佳结果
    print(f"Best Acc after optimization: {best_acc}")
    return best_result

def print_evaluation_results(results):
    # 打印结果
    for mode_name, result in results.items():
        print(
            f'View: {mode_name}(post-processed): '
            f'Best Acc: {round(result["best_acc"], 2)}% | Threshold: {round(result["best_threshold"], 5)} | AUC: {round(result["AUC"], 4)} | '
            f'Normal Acc: {round(result["normal_acc"], 2)}% | Anormal Acc: {round(result["anormal_acc"], 2)}% | '
            f'Normal Precision: {round(result["normal_precision"], 2)} | Anormal Precision: {round(result["anormal_precision"], 2)} |' 
            #f' Score Distribution: {result["score_distribution"]} |'
            #f' Ratio: {result["ratio"]} |'
            f'Normal Ratio: {result["best_normal_ratio"]} | Fuzzy Anomalous Ratio: {result["best_fuzzy_anomalous_ratio"]} | Typical Anomalous Ratio: {result["best_typical_anomalous_ratio"]}'
        )

def evaluate_scores(score_folder, hashmap, gt, weights):
    # 存储结果的字典
    results = {}

    # 遍历hashmap中的每个模式
    for mode, mode_name in hashmap.items():
        # 获取融合得分
        #score = get_fusion_score(score_folder, mode)
        score = get_fusion_score_weighted(score_folder, mode, weights)

        # 后处理得分
        score = post_process(score, 6)

        # 评估得分
        best_acc, best_threshold, AUC, normal_acc, anormal_acc, normal_precision, anormal_precision, best_normal_ratio, best_fuzzy_anomalous_ratio, best_typical_anomalous_ratio = evaluate(score,gt,False)

        # 存储结果
        results[mode_name] = {
            'best_acc': best_acc,
            'best_threshold': best_threshold,
            'AUC': AUC,
            'normal_acc': normal_acc,
            'anormal_acc': anormal_acc,
            'normal_precision': normal_precision,
            'anormal_precision': anormal_precision,
            # 'score_distribution': score_distribution,
            # 'ratio': ratio
            'best_normal_ratio': best_normal_ratio,
            'best_fuzzy_anomalous_ratio': best_fuzzy_anomalous_ratio,
            'best_typical_anomalous_ratio': best_typical_anomalous_ratio
        }

    return results

best_result = optimize_weights(score_folder, hashmap, gt, max_iterations=20, weight_increment=0.01)
print_evaluation_results(best_result)

# for mode, mode_name in hashmap.items():
#     score = get_fusion_score(score_folder, mode)
#
#     # Post-process score
#     score = post_process(score, 6)
#
#     # Evaluate after post-processing
#     best_acc, best_threshold, AUC, normal_acc, anormal_acc, normal_precision, anormal_precision = evaluate(score, gt, False)
#     print(
#         f'View: {mode_name}(post-processed): '
#         f'Best Acc: {round(best_acc, 2)}% | Threshold: {round(best_threshold, 5)} | AUC: {round(AUC, 4)} | '
#         f'Normal Acc: {round(normal_acc, 2)}% | Anormal Acc: {round(anormal_acc, 2)}% | '
#         f'Normal Precision: {round(normal_precision, 2)} | Anormal Precision: {round(anormal_precision, 2)}'
#     )

#
# for mode, mode_name in hashmap.items():
#     #score = get_score(score_folder, mode)
#     score = get_fusion_score(score_folder, mode)
#
#     # Evaluate before post-processing
#     best_acc, best_threshold, AUC, normal_acc, anormal_acc, normal_precision, anormal_precision = evaluate(score, gt, False)
#     print(
#         f'Mode: {mode_name}: '
#         f'Best Acc: {round(best_acc, 2)}% | Threshold: {round(best_threshold, 5)} | AUC: {round(AUC, 4)} | '
#         f'Normal Acc: {round(normal_acc, 2)}% | Anormal Acc: {round(anormal_acc, 2)}% | '
#         f'Normal Precision: {round(normal_precision, 2)} | Anormal Precision: {round(anormal_precision, 2)}'
#     )
#
#     # Post-process score
#     score = post_process(score, 6)
#
#     # Determine whether to plot based on mode
#     if mode == 'fusion_all':  # Only plot for 'fusion_all'
#         plot_flag = True
#     else:
#         plot_flag = False
#
#     # Evaluate after post-processing
#     best_acc, best_threshold, AUC, normal_acc, anormal_acc, normal_precision, anormal_precision = evaluate(score, gt, plot_flag)
#     print(
#         f'View: {mode_name}(post-processed): '
#         f'Best Acc: {round(best_acc, 2)}% | Threshold: {round(best_threshold, 5)} | AUC: {round(AUC, 4)} | '
#         f'Normal Acc: {round(normal_acc, 2)}% | Anormal Acc: {round(anormal_acc, 2)}% | '
#         f'Normal Precision: {round(normal_precision, 2)} | Anormal Precision: {round(anormal_precision, 2)}'
#     )


# for mode, mode_name in hashmap.items():
#     score = get_score(score_folder, mode)
#     best_acc, best_threshold, AUC, normal_acc, anormal_acc = evaluate(score, gt, True)
#     print(
#         f'Mode: {mode_name}: Best Acc: {round(best_acc, 2)}% | Threshold: {round(best_threshold, 2)} | AUC: {round(AUC, 4)} | Normal Acc: {round(normal_acc, 2)}% | Anormal Acc: {round(anormal_acc, 2)}%'
#     )
#     score = post_process(score, 6)
#     best_acc, best_threshold, AUC, normal_acc, anormal_acc = evaluate(score, gt, True)
#     print(
#         f'View: {mode_name}(post-processed): : Best Acc: {round(best_acc, 2)}% | Threshold: {round(best_threshold, 2)} | AUC: {round(AUC, 4)} | Normal Acc: {round(normal_acc, 2)}% | Anormal Acc: {round(anormal_acc, 2)}%'
#     )

# for mode, mode_name in hashmap.items():
#     score = get_score(score_folder, mode)
#     best_acc, best_threshold, AUC = evaluate(score, gt, False)
#     print(
#         f'Mode: {mode_name}:      Best Acc: {round(best_acc, 2)} | Threshold: {round(best_threshold, 2)} | AUC: {round(AUC, 4)}')
#     score = post_process(score, 6)
#     best_acc, best_threshold, AUC = evaluate(score, gt, False)
#     print(
#         f'View: {mode_name}(post-processed):       Best Acc: {round(best_acc, 2)} | Threshold: {round(best_threshold, 2)} | AUC: {round(AUC, 4)} \n')