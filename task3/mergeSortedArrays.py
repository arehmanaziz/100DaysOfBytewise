def mergeArray(array1, array2):
    mergedArray = []
    i = 0
    j = 0
    while i < len(array1) and j < len(array2):
        if array1[i] < array2[j]:
            mergedArray.append(array1[i])
            i += 1
        else:
            mergedArray.append(array2[j])
            j += 1

    while i < len(array1):
        mergedArray.append(array1[i])
        i += 1
        
    while j < len(array2):
        mergedArray.append(array2[j])
        j += 1

    return mergedArray

if __name__ == '__main__':
    array1 = [1, 3, 5]
    array2 = [2, 4, 6]

    mergedArray = mergeArray(array1, array2)
    print(mergedArray)