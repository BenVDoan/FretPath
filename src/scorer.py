# calculates the amount of matched notes by checking the longest common subsequence
def lcs_length(expected, detected):
    m, n = len(expected), len(detected)
    dp = [[0] * (n + 1) for i in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if expected[i - 1] == detected[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]

def check(expected, detected):
    score = {
        "lcs": lcs_length(expected, detected),
        "n_det": len(detected),
        "n_exp": len(expected)
    }

    return score
def score(lcs, n_det, n_exp):
    recall = lcs / n_exp if n_exp else 0.0
    precision = lcs/ n_det if n_det else 0.0
    F1 = (2 * recall * precision) / (recall + precision) if lcs else 0.0
    return recall, precision, F1