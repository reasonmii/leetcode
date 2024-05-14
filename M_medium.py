
# ======================================================================
# 2. Add Two Numbers
# Topic : ListNode
# ======================================================================

class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """

        carry = 0
        root = n = ListNode(0)

        while l1 or l2 or carry:
            v1 = v2 = 0
            if l1:
                v1 = l1.val
                l1 = l1.next
            if l2:
                v2 = l2.val
                l2 = l2.next

            # divmod(numerator, denominator)
            # divmod(7, 10) = 0, 7
            # divmod(10, 10) = 1, 0
            # divmod(8, 10) = 0, 8
            carry, val = divmod(v1+v2+carry, 10)
            n.next = ListNode(val)
            n = n.next

        return root.next

# ======================================================================
# 3. Longest Substring Without Repeating Characters
# Topic : dictionary, string
# ======================================================================

class Solution(object):
    def lengthOfLongestSubstring(self, s):

        len_s = 0
        idx = {}

        i = 0
        for j in range(len(s)):
            if s[j] in idx:
                i = max(i, idx[s[j]]) # max(idx[a], 4) = max(1, 4) = 4
            len_s = max(len_s, j-i+1) # 1 / 2 / 3 / max(3, 4-4+1) = 3
            idx[s[j]] = j+1 # idx[a] = 1 / idx[b] = 2 / idx[c] = 3

        return int(len_s)

# ======================================================================
# 5. Longest Palindromic Substring
# Topic : string, dynamic programming
# ======================================================================

class Solution(object):
    def longestPalindrome(self, s):        
        rst = ""
        dp = [[0]*len(s) for _ in range(len(s))]

        for i in range(len(s)):
            dp[i][i] = True
            # [[True, 0, 0, 0, 0],
            # [0, True, 0, 0, 0],
            # [0, 0, True, 0, 0],
            # [0, 0, 0, True, 0],
            # [0, 0, 0, 0, True]]
            rst = s[i]

        # babad
        # range(5,5) -> None
        for i in range(len(s)-1,-1,-1): # 4 -> 3 -> 2 -> 1 -> 0
            for j in range(i+1, len(s)):  # None -> 4 -> 3~4 -> 2~4 -> 1~4
                if s[i] == s[j]: # i=1, j=3
                    if j-i == 1 or dp[i+1][j-1] is True: # dp[2][2]
                        dp[i][j] = True # dp[1][3]
                        if len(rst) < len(s[i:j+1]): # len(s[1:4]) = 3
                            rst = s[i:j+1]

        # [[True, 0, True, 0, 0],
        # [0, True, 0, True, 0],
        # [0, 0, True, 0, 0],
        # [0, 0, 0, True, 0],
        # [0, 0, 0, 0, True]]
                
        return rst
        
# ======================================================================
# 6. Zigzag Conversion
# Topic : string
# ======================================================================

class Solution(object):
    def convert(self, s, numRows):
        if numRows == 1:
            return s

        rst = ["" for _ in range(numRows)] # ['', '', '']

        chk = True
        k = 0

        for i in range(len(s)):

            if chk:
                rst[k] += s[i]
                k += 1
                if k == numRows:
                    chk = False
                    k = -2   # 2nd last one
            else:
                rst[k] += s[i]
                k -= 1
                if k == -numRows-1:
                    chk = True
                    k = 1
            
        return "".join(rst)

        # ['P', '', '', '']
        # ['P', 'A', '', '']
        # ['P', 'A', 'Y', '']
        # ['P', 'A', 'Y', 'P']
        # ['P', 'A', 'YA', 'P']
        # ['P', 'AL', 'YA', 'P']
        # ['PI', 'AL', 'YA', 'P']
        # ['PI', 'ALS', 'YA', 'P']
        # ['PI', 'ALS', 'YAH', 'P']
        # ['PI', 'ALS', 'YAH', 'PI']
        # ['PI', 'ALS', 'YAHR', 'PI']
        # ['PI', 'ALSI', 'YAHR', 'PI']
        # ['PIN', 'ALSI', 'YAHR', 'PI']
        # ['PIN', 'ALSIG', 'YAHR', 'PI']

# ======================================================================
# 7. Reverse Integer
# Topic : mod
# ======================================================================

class Solution(object):
    def reverse(self, x):

        neg = False if x >= 0 else True
        x = abs(x)
        rst = 0

        while x != 0:
            digit = x % 10 # 3, 2, 1
            rst = rst * 10 + digit
            x //= 10

        if rst >= 2**31 - 1 or rst < -2**31:
            return 0

        if neg:
            rst *= -1

        return rst

# ======================================================================
# 11. Container With Most Water
# Topic : while
# ======================================================================

class Solution(object):
    def maxArea(self, height):
        rst, i, j = 0, 0, len(height)-1

        while i < j:
            w = j - i
            if height[i] <= height[j]:
                rst = max(rst, w * height[i])
                i += 1
            else:
                rst = max(rst, w * height[j])
                j -= 1
        return rst

# ======================================================================
# 12. Integer to Roman
# Topic : string
# ======================================================================

class Solution(object):
    def intToRoman(self, num):

        rst = ""
        rom = [[1000, 'M'], [900, 'CM'], [500, 'D'], [400, 'CD'], [100, 'C'], \
        [90, 'XC'], [50, 'L'], [40, 'XL'], [10, 'X'], [9, 'IX'], [5, 'V'], [4, 'IV'], [1, 'I']]

        for i in range(len(rom)):
            while num >= rom[i][0]:
                rst += rom[i][1]
                num -= rom[i][0]

        return rst

# ======================================================================
# 15. 3Sum
# Topic : set, add, tuple(sorted([]))
# ======================================================================

class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """

        rst = set() # [-1, 0, 1], [0, 1, -1] --> [-1, 0, 1]

        # neg, pos, zero
        n, p, z = [], [], []
        for num in nums:
            if num < 0:
                n.append(num)
            elif num > 0:
                p.append(num)
            else:
                z.append(num)

        N, P = set(n), set(p)

        # case 1) at least 1 zero in the list
        # 0, a, -a : 0 +1 -1 = 0
        if z:
            for num in P:
                if -1 * num in N:
                    rst.add((-1*num, 0, num))

        # case 2) at least 3 zeros
        if len(z) >= 3:
            rst.add((0, 0, 0))

        # case 3) 2 negatives + 1 positive
        # -3 -1 +4 = 0
        for i in range(len(n)):
            for j in range(i+1, len(n)):
                target = -1 * (n[i] + n[j])
                if target in P:       ### p 이 아니라 P
                    rst.add(tuple(sorted([n[i], n[j], target])))
                    # [-1, -2, 3] == [-2, -1, 3]

        # case 4) 2 positives + 1 negative
        for i in range(len(p)):
            for j in range(i+1, len(p)):
                target = -1 * (p[i] + p[j])
                if target in N:            ### n 이 아니라 N
                    rst.add(tuple(sorted([p[i], p[j], target])))

        return rst

# ======================================================================
# 17. Letter Combinations of a Phone Number
# Topic : 
# ======================================================================








