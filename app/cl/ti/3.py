class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        char_index = {}  # 记录字符最近一次出现的下标
        left = 0
        max_len = 0
        max_sub = ""
        #pwwkew
        for right, char in enumerate(s): 
            # 如果字符已存在，并且在当前窗口内，就收缩左边界
            if char in char_index and char_index[char] >= left:
                left = char_index[char] + 1 
            # 更新字符的最新位置
            char_index[char] = right 
            # 计算当前窗口长度
            curr_len = right - left + 1
            if curr_len > max_len:
                max_len = curr_len
                max_sub = s[left:right + 1]
        
        return max_len, max_sub
    def fastlengthOfLongestSubstring(self, s: str) -> int:
        dic, res, left = {}, 0, -1
        for right in range(len(s)):
            if s[right] in dic:
                left = max(dic[s[right]], left) # 更新左指针 i
            dic[s[right]] = right # 哈希表记录
            res = max(res, right - left) # 更新结果
        return res

if __name__ == '__main__':
    s=Solution()
    #print(s.lengthOfLongestSubstring("abcabcbb"))
    #print(s.lengthOfLongestSubstring("bbbbb"))
    print(s.lengthOfLongestSubstring("pwwkew"))
    #print(s.lengthOfLongestSubstring(""))
       
