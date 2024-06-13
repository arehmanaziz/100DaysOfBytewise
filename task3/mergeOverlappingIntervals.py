def mergeIntervals(intervals):
    if not intervals:
        return []

    intervals.sort(key=lambda x: x[0])

    merged = [intervals[0]]
    
    for i in intervals[1:]:
        if merged[-1][1] >= i[0]:
            merged[-1][1] = max(merged[-1][1], i[1])
        else:
            merged.append(i)

    return merged

arr = [[1, 3], [2, 6], [8, 10], [15, 18]]
merged_intervals = mergeIntervals(arr)
print("The Merged Intervals are:", merged_intervals)

arr2 = [[6, 8], [1, 9], [2, 4], [4, 7]]
merged_intervals2 = mergeIntervals(arr2)
print("The Merged Intervals are:", merged_intervals2)
