
# ======================================================================
# 1. Two Sum
# Topic : dictionary
# ======================================================================

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:

        dic = {}
        for i, n in enumerate(nums):
            diff = target - n
            if diff in dic:
                return [dic[diff], i]
            dic[n] = i

# ======================================================================
# 9. Palindrome Number
# Topic : string
# ======================================================================

class Solution(object):
    def isPalindrome(self, x):
        if x < 0 or (x % 10 == 0 and x != 0):
            return False
        
        half = 0
        while x > half:
            half = half * 10 + x % 10 # 1 -> 12
            x = x // 10 # 12 -> 1
        
        return (x == half) or (x == half//10)

# ======================================================================
# 13. Roman to Integer
# Topic : string
# ======================================================================

class Solution(object):
    def romanToInt(self, s):
        dic = {'I':1, 'V':5, 'X':10, 'L':50, 'C':100, 'D':500, 'M':1000}

        s = s.replace("IV", "IIII")
        s = s.replace("IX", "VIIII")
        s = s.replace("XL", "XXXX")
        s = s.replace("XC", "LXXXX")
        s = s.replace("CD", "CCCC")
        s = s.replace("CM", "DCCCC")

        rst = 0
        for ch in s:
            rst += dic[ch]        

        return rst        

# ======================================================================
# 20. Valid Parentheses
# Topic : string, brackets, stack
# ======================================================================

class Solution(object):
    def isValid(self, s):
        stack = []

        for char in s:
            if char in "([{":
                stack.append(char)
            else:
                if stack == [] or \
                    (char == ")" and stack[-1] != "(") or \
                    (char == "]" and stack[-1] != "[") or \
                    (char == "}" and stack[-1] != "{"):
                    return False
                stack.pop()
        
        return not stack # stack is empty

# ======================================================================
# 21. Merge Two Sorted Lists
# Topic : LinkedList
# ======================================================================

class Solution(object):
    def mergeTwoLists(self, list1, list2):

        cur = dummy = ListNode()
        
        while list1 and list2:
            if list1.val <= list2.val:
                cur.next = list1
                list1, cur = list1.next, list1
            else:
                cur.next = list2
                list2, cur = list2.next, list2

        if list1 or list2:
            cur.next = list1 if list1 else list2

        return dummy.next

# ======================================================================
# 28. Find the Index of the First Occurrence in a String
# Topic : string, index
# ======================================================================

class Solution(object):
    def strStr(self, haystack, needle):
        
        end = len(needle)
        for start in range(len(haystack)):
            if haystack[start:start+end] == needle:
                return start
        return -1

# ======================================================================
# 66. Plus One
# Topic : list, plus
# ======================================================================

class Solution(object):
    def plusOne(self, digits):

        for i in reversed(range(len(digits))):
            if digits[i] != 9:
                digits[i] += 1
                return digits
            digits[i] = 0

        return [1] + digits

# ======================================================================
# 70. Climbing Stairs
# Topic : Dynamic Programming
# ======================================================================

class Solution(object):
    def climbStairs(self, n):

        if n == 1:
            return 1
        
        dp = [0 for _ in range(n+1)]
        dp[1], dp[2] = 1, 2

        for i in range(3, n+1):
            dp[i] = dp[i-2] + dp[i-1]

        return dp[-1]















