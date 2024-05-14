
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
# Topic : dfs
# ======================================================================

class Solution(object):
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """

        dic = {"2":"abc", "3":"def", "4":"ghi", "5":"jkl", "6":"mno", "7":"pqrs", "8":"tuv", "9":"wxyz"}

        rst = []

        if len(digits) == 0:
            return rst

        self.dfs(digits, 0, dic, '', rst) # "23"
        return rst
    
    def dfs(self, nums, idx, dic, path, rst):

        if idx >= len(nums):
            rst.append(path) # "ad"
            return

        strings = dic[nums[idx]] # dic[2] => dic[3]
        for i in strings: # a -> b -> c => d -> e -> f
            self.dfs(nums, idx+1, dic, path+i, rst)
            # "23", 1, dic, "a", []
            # "23", 2, dic, "ad", []

# ======================================================================
# 18. 4Sum
# Topic : while, append
# ======================================================================

class Solution(object):
    def fourSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[List[int]]
        """

        nums.sort()
        N = len(nums)
        rst = []

        for i in range(N):
            if i > 0 and nums[i] == nums[i-1]:
                continue # [2, 2, 2, 2] -> duplicates!

            for j in range(i+1, N):
                if j > i+1 and nums[j] == nums[j-1]:
                    continue # duplicates!

                x = target - nums[i] - nums[j] # new target
                s, e = j+1, N-1 # start, end
                while s < e:
                    if nums[s] + nums[e] == x:
                        rst.append([nums[i], nums[j], nums[s], nums[e]])
                        s += 1
                        while s < e and nums[s] == nums[s-1]:
                            s += 1 # duplicates!
                    elif nums[s] + nums[e] < x:
                        s += 1
                    else: # nums[s] + nums[e] > x
                        e -= 1

        return rst

# ======================================================================
# 19. Remove Nth Node From End of List
# Topic : LinkedList, fast and slow
# ======================================================================

class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """

        fast, slow = head, head

        for _ in range(n): fast = fast.next # 2 -> 3
 
        # print(head.val, slow.val, fast.val) # 1, 1, 3

        if not fast:
            return head.next

        while fast.next:
            fast = fast.next # 4 -> 5
            slow = slow.next # 2 -> 3

        slow.next = slow.next.next # 5
        return head

# ======================================================================
# 22. Generate Parentheses
# Topic : dfs, string
# ======================================================================

class Solution(object):
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """

        def dfs(left, right, s):
            if len(s) == n*2:
                rst.append(s)
                return
            if left < n:
                dfs(left+1, right, s+'(')
            if right < left:
                dfs(left, right+1, s+')')
            
        rst = []
        dfs(0, 0, '')
        return rst

# ======================================================================
# 31. Next Permutation
# Topic : sort, reverse
# ======================================================================

class Solution(object):
    def nextPermutation(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """

        # [1,2,3] // [1,3,2]
        for i in range(len(nums)-1, 0, -1):   # 2 // 2 -> 1
            if nums[i-1] < nums[i]:           # nums[1] < nums[2] // nums[0] < nums[1]
                nums[i:] = sorted(nums[i:])   # [1,2,3] // [1,2,3]

                j = i - 1                     # 1 // 0

                for k in range(i, len(nums)):               # 2 // 1
                    if nums[j] < nums[k]:                   # nums[1] < nums[2] // nums[0] < nums[1]
                        nums[k], nums[j] = nums[j], nums[k] # [1,3,2] // [2,1,3]
                        return nums

        return nums.reverse()

# ======================================================================
# 33. Search in Rotated Sorted Array
# Topic : Array, mid, while
# ======================================================================

class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """

        n = len(nums)
        left, right = 0, n -1

        while left <= right:
            mid = (left+right) // 2

            if nums[mid] == target:
                return mid

            if nums[left] <= nums[mid]:
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
        
        return -1

# ======================================================================
# 36. Valid Sudoku
# Topic : Array
# ======================================================================

class Solution(object):
    def isValidSudoku(self, board):
        
        seen = []
        for i, row in enumerate(board):
            for j, c in enumerate(row):
                if c != '.':
                    seen += [(c, j), (i, c), (i//3, j//3, c)]

        return len(seen) == len(set(seen))

# ======================================================================
# 39. Combination Sum
# Topic : dfs
# ======================================================================

class Solution(object):
    def combinationSum(self, candidates, target):
        rst = []
        candidates.sort

        def dfs(target, idx, path):
            if target < 0:
                return
            if target == 0:
                rst.append(path)
                return
            for i in range(idx, len(candidates)):
                dfs(target - candidates[i], i, path+[candidates[i]])

        dfs(target, 0, [])
        return rst

# ======================================================================
# 43. Multiply Strings
# Topic : string to number (ord), decode, encode
# ======================================================================

class Solution(object):
    def multiply(self, num1, num2):

        if num1 == '0' or num2 == '0':
            return '0'
        
        def decode(word):
            num = 0
            for w in word:
                num = num *10 + (ord(w) - ord('0'))
            return num
        
        def encode(num):
            word = ''
            while num:
                n = num % 10
                num //= 10
                word = chr(ord('0') + n) + word
            return word

        rst = decode(num1) * decode(num2)
        return encode(rst)
        
# ======================================================================
# 48. Rotate Image
# Topic : Array, rotate
# ======================================================================

class Solution(object):
    def rotate(self, matrix):
        matrix.reverse()

        for i in range(len(matrix)):
            for j in range(i):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

        return matrix

# ======================================================================
# 49. Group Anagrams
# Topic : dic, join
# ======================================================================

class Solution(object):
    def groupAnagrams(self, strs):
        dic = {}
        rst = []

        for s in strs:
            word = ''.join(sorted(s))    # aet (eat) -> aet (tea)
            if word in dic:              
                rst[dic[word]].append(s) # rst[0].append('tea')
            else:
                dic[word] = len(rst) # dic['aet'] = 0
                rst.append([s])      # ['eat']

        return rst

# ======================================================================
# 53. Maximum Subarray
# Topic : Array, max, dynamic programming
# ======================================================================

class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        # [5,4,-1,7,8]
        max_v = nums[0] # 5
        arr = [max_v]   # [5]

        for i in range(1, len(nums)): # 1 -> 2 -> 3 -> 4
            val = max(arr[i-1]+nums[i], nums[i]) # max(5+4, 4) -> max(9-1, -1) -> max(8+7, 7)
            arr.append(val) # [5,9], [5,9,8], [5,9,8,15]

            if arr[i] > max_v: # 9 > 5 -> 8 > 9 -> 15 > 9
                max_v = arr[i] # 9 -> 9 -> 15
        return max_v

# ======================================================================
# 54. Spiral Matrix
# Topic : matrix, pop, append
# ======================================================================

class Solution(object):
    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """

        rst = []

        while matrix:

            # rst : [1,2,3] -> [1,2,3,6,9,8,7,4,5]
            # matrix : [[4,5,6],[7,8,9]]
            rst += matrix.pop(0)

            # rst : [1,2,3,6] -> [1,2,3,6,9]
            # matrix : [[4,5], [7,8,9]] -> [[4,5], [7,8]]
            if matrix and matrix[0]:
                for row in matrix:
                    rst.append(row.pop())

            # rst : [1,2,3,6,9,8,7]
            # matrix : [[4,5]]
            if matrix:
                rst += matrix.pop()[::-1]

            # rst : [1,2,3,6,9,8,7,4]
            # matrix[[5]]
            if matrix and matrix[0]:
                for row in matrix[::-1]:
                    rst.append(row.pop(0))

        return rst

# ======================================================================
# 55. Jump Game
# Topic : Array
# ======================================================================

class Solution(object):
    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """

        last = len(nums)-1 # 4

        for i in range(len(nums)-2, -1, -1): # 3 -> 2 -> 1 -> 0
            # 3+nums[3]=3+0=3 >= 4
            # 2+nums[2]=2+1=3 >= 4
            # 1+nums[1]=1+2=3 >= 4
            # 0+nums[0]=0+3=3 >= 4
            if i + nums[i] >= last:
                last = i
        return last == 0

# ======================================================================
# 56. Merge Intervals
# Topic : Array, sort (lambda)
# ======================================================================

class Solution(object):
    def merge(self, intervals):        
        itv = sorted(intervals, key=lambda x: x[0]) ###
        rst = []

        for arr in itv:
            if rst and arr[0] <= rst[-1][1]:
                rst[-1][1] = max(rst[-1][1], arr[1])
            else:
                rst += [arr]
        return rst

# ======================================================================
# 62. Unique Paths
# Topic : dynamic programming
# ======================================================================

class Solution(object):
    def uniquePaths(self, m, n):
        dp = [[1] * n] * m

        for r in range(1, m):
            for c in range(1, n):
                dp[r][c] = dp[r-1][c] + dp[r][c-1]

        return dp[-1][-1]

# ======================================================================
# 64. Minimum Path Sum
# Topic : dynamic programming
# ======================================================================

class Solution(object):
    def minPathSum(self, grid):
    
        # [[1,2,3],[4,5,6]]
        m, n = len(grid), len(grid[0]) # 2, 3

        # [[1,3,6],[4,5,6]]
        for i in range(1, n):
            grid[0][i] += grid[0][i-1]

        # [[1,3,6],[5,5,6]]
        for i in range(1,m):
            grid[i][0] += grid[i-1][0]

        for i in range(1,m):
            for j in range(1,n):
                grid[i][j] += min(grid[i-1][j], grid[i][j-1])

        return grid[-1][-1]















