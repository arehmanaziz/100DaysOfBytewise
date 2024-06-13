def min_edit_distance(str1, str2):
    m, n = len(str1), len(str2)
    
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j  # If str1 is empty, insert all characters of str2
            elif j == 0:
                dp[i][j] = i  # If str2 is empty, remove all characters of str1
            elif str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # Characters match, no operation needed
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],    # Remove
                                   dp[i][j - 1],    # Insert
                                   dp[i - 1][j - 1])  # Replace

    return dp[m][n]


str1 = "kitten"
str2 = "sitting"
result = min_edit_distance(str1, str2)
print(f"The minimum number of operations required to transform '{str1}' into '{str2}' is {result}")
