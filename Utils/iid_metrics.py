import numpy as np

# Identical Distribution

# ----------------------Kolmogorov-Smirnov test-----------------------
# statistic->0, p_value->1
from scipy.stats import kstest
def perform_ks_test(subset, overall_data):
    flat_subset = np.array([np.array(item).flatten() for item, _ in subset])
    flat_overall = np.array([np.array(item).flatten() for subset in overall_data for item, _ in subset])
    statistic_results = []
    p_value_results = []
    for i in range(flat_subset.shape[1]):
        statistic, p_value = kstest(flat_subset[:, i], flat_overall[:, i])
        statistic_results.append(statistic)
        p_value_results.append(p_value)
    ks_results = (np.mean(statistic_results), np.mean(p_value_results))
    return ks_results

def KS_test(dataset):
    ks_test_results = []
    for i in range(len(dataset)):
        ks_test_results.append(perform_ks_test(dataset[i], dataset))
    # for i, result in enumerate(ks_test_results):
    #     print(f"Subset {i+1} - Statistic: {result[0]}, P-value: {result[1]}")
    ks_test_results = np.mean(np.array(ks_test_results), axis=0)
    return ks_test_results


# Independent

# ------------------------chi-squared test-----------------------
from scipy.stats import chi2_contingency
# to do

# -----------------------mutual_info_score-----------------------
# 互信息的取值范围是从0到正无穷，其中0表示两个变量之间没有相互依赖关系，
# 而较大的值表示两个变量之间存在较强的相互依赖关系。
from sklearn.metrics import mutual_info_score
def compute_mutual_info(subset):
    n = len(subset)
    mi_values = []
    for i in range(n):
        for j in range(i + 1, n):
            flat_subset_i = np.array([np.array(item) for item, _ in subset[i]]).flatten()
            flat_subset_j = np.array([np.array(item) for item, _ in subset[j]]).flatten()
            min_length = min(len(flat_subset_i), len(flat_subset_j))
            flat_subset_i = flat_subset_i[:min_length]
            flat_subset_j = flat_subset_j[:min_length]
            mi = mutual_info_score(flat_subset_i, flat_subset_j)
            mi_values.append(mi)
    average_mi_subset = np.mean(mi_values) if mi_values else 0

    mi_values = []
    total_mi_inside = []
    # for i in range(n):
    #     flat_subsets = np.array([np.array(item).flatten() for item, _ in subset[i]])
    #     for j in range(len(flat_subsets)):
    #         for k in range(j+1, len(flat_subsets)):
    #             mi = mutual_info_score(flat_subsets[j], flat_subsets[k])
    #             # print(j, ' mi: ', mi)
    #             mi_values.append(mi)
    #     average_mi_inside = np.mean(mi_values) if mi_values else 0
    #     print('average_mi_inside: ', average_mi_inside)
    #     total_mi_inside.append(average_mi_inside)

    return average_mi_subset, total_mi_inside


# -----------------------Pearson correlation coefficient----------------------
# 相关系数接近0通常表明两个变量之间无线性关系，而不是统计上的独立性。
# correlations: +1: 完全正相关; -1: 完全负相关。0: 无线性相关。
# |correlation| > 0.7 ：强相关；在 0.5 到 0.7 之间：中等强度的相关；< 0.5 ：弱相关。
# p-value < 0.05，则拒绝零假设，认为两个变量之间存在统计学上显著的相关性。
from scipy.stats import pearsonr
def Pearson_correlation(subset):
    n = len(subset)
    correlations = []
    p_values = []
    for i in range(n):
        for j in range(i + 1, n):
            flat_subset_i = np.array([np.array(item) for item, _ in subset[i]]).flatten()
            flat_subset_j = np.array([np.array(item) for item, _ in subset[j]]).flatten()
            min_length = min(len(flat_subset_i), len(flat_subset_j))
            flat_subset_i = flat_subset_i[:min_length]
            flat_subset_j = flat_subset_j[:min_length]
            correlation, p_value = pearsonr(flat_subset_i, flat_subset_j)
            correlations.append(correlation)
            p_values.append(p_value)
    correlations = np.mean(correlations) if correlations else 0
    p_values = np.mean(p_values) if p_values else 0
    return correlations, p_values


# ------------Cross-Correlation Analysis & Granger Causality Test-------------
# 交叉相关性分析可以用来评估两个时间序列在不同时间滞后下的相关性。
# 如果所有时间滞后的相关系数都非常接近于零，那么可以认为这两个时间序列是独立的。
from statsmodels.tsa.stattools import ccf
# Granger 因果检验可以用来检验一个时间序列是否能预测另一个时间序列。
# 如果p-value低于特定的阈值（通常是0.05或0.01），则可以认为有足够的证据拒绝原假设，
# 即一个序列在统计上是另一个序列的Granger原因。
from statsmodels.tsa.stattools import grangercausalitytests
def Cross_Correlation(subset, feature_size = 6):
    n = len(subset)
    cross_corrs = []
    grangers = []
    for i in range(n):
        for j in range(i + 1, n):
            flat_subset_i = np.array([np.array(item) for item, _ in subset[i]]).reshape(-1, feature_size)
            flat_subset_j = np.array([np.array(item) for item, _ in subset[j]]).reshape(-1, feature_size)
            min_length = min(len(flat_subset_i), len(flat_subset_j))
            flat_subset_i = flat_subset_i[:min_length]
            flat_subset_j = flat_subset_j[:min_length]
            cross_corr_one = []
            for feature_i in range(feature_size):
                ts1 = flat_subset_i[:, feature_i]
                ts2 = flat_subset_j[:, feature_i]
                # Cross-Correlation Analysis
                cross_corr = ccf(ts1, ts2, adjusted=True)
                cross_corr_one.append(cross_corr)
                # Granger Causality Test
                data = np.column_stack([ts1, ts2])
                result = grangercausalitytests(data, maxlag=2, verbose=False)
                grangers.append(result)
            cross_corrs.append(np.mean(cross_corr_one))
    cross_corrs = np.mean(cross_corrs) if cross_corrs else 0

    grangers_p_values = []
    for i, granger in enumerate(grangers):
        p_values = [round(granger[lag][0]['ssr_chi2test'][1], 4) for lag in range(1, 3)]
        # print(f"Feature {i+1} Granger Causality P-values:", p_values)
        grangers_p_values.append(np.mean(p_values))
    grangers_p_values = np.mean(grangers_p_values) if grangers_p_values else 0

    return cross_corrs, grangers_p_values