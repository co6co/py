class Solution:
    def twoSum(self, nums: list[int], target: int) -> list[int]:
        # 基本方法 双层循环，时间复杂度 O(n²)、空间 O(1)
        for i,v in enumerate(nums):
            for j in range(i+1,len(nums)):
                if v+nums[j]==target:
                    return [i,j]

        # 方案2 时间 O(n)，空间 O(n)，用空间换时间
        seen = {} # 值 -> 下标 
        for i, v in enumerate (nums): 
            if target - v in seen: 
                return [seen[target - v], i]
            seen[v] = i
        

s='123'
s=list(s)
s.reverse() 
int(''.join(s))
print(int(''.join(s)))
