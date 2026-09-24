def merge_sort(arr1: list, arr2: list):
    """
    双指针排序
    :arr1 已经排序的数组
    :arr2 已经排序的数组
    :return: 合并后的数组
    """
    merge_result=[]
    i=j=0
    while i<len(arr1) and j<len(arr2):
        if arr1[i]<=arr2[j]:
            merge_result.append(arr1[i])
            i+=1
        else:
            merge_result.append(arr2[j])
            j+=1
    merge_result.extend(arr1[i:])
    merge_result.extend(arr2[j:])
    return merge_result


if __name__ =='__main__':
    arr1=['a',"b",'c','d']
    arr2=['e','f','g','h']
    print(merge_sort(arr1,arr2))