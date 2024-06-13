def maxSubArray(nums):
    if not nums:
        return 0

    max_current = max_global = nums[0]

    for num in nums[1:]:
        max_current = max(num, max_current + num)
        if max_current > max_global:
            max_global = max_current

    return max_global


input_array = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
max_sum = maxSubArray(input_array)
print(f"The maximum sum of a contiguous subarray is {max_sum}")
