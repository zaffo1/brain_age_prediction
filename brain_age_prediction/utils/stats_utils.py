'''
Useful statistical tools
'''
import numpy as np
from scipy.stats import pearsonr

def empirical_p_value(group1, group2, num_permutations=100000):
    '''
    Calculate the empirical p-value for the difference in means between two groups using permutation testing.

    :param array-like group1: Data for the first group.
    :param array-like group2: Data for the second group.
    :param int num_permutations: Number of permutations to perform
                                 for the permutation test. Default is 100,000.

    :return: Empirical p-value for the observed difference in means.
    :rtype: float

    This function performs a permutation test to
    estimate the empirical p-value for the difference in means between two groups.
    The observed test statistic is the difference in means between group2 and group1.

    The function generates permuted test statistics by randomly permuting
    the data between the two groups and calculates the difference in means for each permutation.
    The empirical p-value is then calculated as the proportion
    of permuted differences in means that are greater than or equal to the observed difference in means.
    '''

    # Observed test statistic (difference in means)
    observed_statistic = np.mean(group2) - np.mean(group1)

    # Initialize array to store permuted test statistics
    permuted_statistics = np.zeros(num_permutations)

    # Perform permutations and calculate permuted test statistics
    for i in range(num_permutations):
        # Concatenate and randomly permute the data
        combined_data = np.concatenate((group1, group2))
        np.random.shuffle(combined_data)

        # Split permuted data into two groups
        permuted_group1 = combined_data[:len(group1)]
        permuted_group2 = combined_data[len(group1):]

        # Calculate test statistic for permuted data
        permuted_statistic = np.mean(permuted_group2) - np.mean(permuted_group1)

        # Store permuted statistic
        permuted_statistics[i] = permuted_statistic

    # Calculate p-value
    p_value = np.sum(np.abs(permuted_statistics) >= np.abs(observed_statistic)) / num_permutations

    print("Empirical p-value:", p_value)

    return p_value


def correlation(x,y,permutation_number=1000):
    '''
    Calculate Pearson correlation coefficient and its p-value between two arrays.

    :param array-like x: First array for correlation.
    :param array-like y: Second array for correlation.
    :param int permutation_number: Number of permutations for computing the empirical p-value. Default is 1000.

    :return: Tuple containing the Pearson correlation coefficient and its empirical p-value.
    :rtype: tuple

    This function calculates the Pearson correlation coefficient (r) between two arrays 'x' and 'y'.
    Additionally, it performs a permutation test to estimate the empirical p-value of the correlation coefficient.
    '''
    r = pearsonr(x,y)[0]
    #Copy one of the features:
    x_s = np.copy(x)
    #Initialize variables:
    permuted_r = []

    p=permutation_number

    for _ in range(0,p):
    #Shuffle one of the features:
        np.random.shuffle(x_s)
        #Computed permuted correlations and store them:
        permuted_r.append(pearsonr(x_s,y)[0])

    #Significance:
    p_val = len(np.where(np.abs(permuted_r)>=np.abs(r))[0])/p
    return r, p_val