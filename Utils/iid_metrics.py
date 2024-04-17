import numpy as np
from scipy.stats import kstest
from scipy.stats import chi2_contingency
from sklearn.metrics import mutual_info_score

# Identical Distribution

# Kolmogorov-Smirnov test: statistic->0, p_value->1

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
    for i, result in enumerate(ks_test_results):
        print(f"Subset {i+1} - Statistic: {result[0]}, P-value: {result[1]}")
    ks_test_results = np.mean(np.array(ks_test_results), axis=0)
    print('ks_test_results: ', ks_test_results)
    return ks_test_results


# Independent

# chi-squared test
# to do

# mutual_info_score
# 互信息的取值范围是从0到正无穷，其中0表示两个变量之间没有相互依赖关系，
# 而较大的值表示两个变量之间存在较强的相互依赖关系。

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
    for i in range(n):
        flat_subsets = np.array([np.array(item).flatten() for item, _ in subset[i]])
        for j in range(len(flat_subsets)):
            for k in range(j+1, len(flat_subsets)):
                mi = mutual_info_score(flat_subsets[j], flat_subsets[k])
                # print(j, ' mi: ', mi)
                mi_values.append(mi)
        average_mi_inside = np.mean(mi_values) if mi_values else 0
        print('average_mi_inside: ', average_mi_inside)
        total_mi_inside.append(average_mi_inside)

    return average_mi_subset, total_mi_inside