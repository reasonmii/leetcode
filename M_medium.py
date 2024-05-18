
# ======================================================================
# 2. Add Two Numbers
# Topic : ListNode
# ======================================================================

class Solution(object):
    def addTwoNumbers(self, l1, l2):

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

# ======================================================================
# 72. Edit Distance
# Topic : dynamic programming
# https://leetcode.com/problems/edit-distance/solutions/3230662/clean-codes-full-explanation-dynamic-programming-c-java-python3/
# ======================================================================

class Solution(object):
    def minDistance(self, word1, word2):

        prev = list(range(len(word2)+1))
        cur = [0] * (len(word2) + 1)

        for i in range(1, len(word1)+1):
            cur[0] = i
            for j in range(1, len(word2)+1):
                if word1[i-1] == word2[j-1]:
                    cur[j] = prev[j-1]
                else:
                    cur[j] = min(prev[j-1] + 1, prev[j] + 1, cur[j-1] + 1)
                
            prev = cur
            cur = [0] * (len(word2) + 1)
        
        return prev[-1]

# ======================================================================
# 74. Search a 2D Matrix
# Topic : mid value of matrix (left, right)
# ======================================================================

class Solution(object):
    def searchMatrix(self, matrix, target):

        # [[1,3,5,7],[10,11,16,20],[23,30,34,60]]
        m, n = len(matrix), len(matrix[0])

        left, right = 0, m*n-1

        while left <= right:
            mid = (left + right) // 2
            mid_val = matrix[mid // n][mid % n] ###
            
            if target == mid_val:
                return True
            elif target < mid_val:
                right = mid-1
            else:
                left = mid+1

        return False

# ======================================================================
# 75. Sort Colors
# Topic : while, swap
# ======================================================================

class Solution(object):
    def sortColors(self, nums):

        red, white, blue = 0, 0, len(nums)-1

        while white <= blue:
            if nums[white] == 0:
                nums[white], nums[red] = nums[red], nums[white]
                white += 1
                red += 1
            elif nums[white] == 1:
                white += 1
            else:
                nums[white], nums[blue] = nums[blue], nums[white]
                blue -= 1

# ======================================================================
# 77. Combinations
# Topic : backtracking
# ======================================================================

class Solution(object):
    def combine(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[List[int]]
        """

        combs = [[]]
        for _ in range(k):
            combs = [[i] + c for c in combs for i in range(1, c[0] if c else n+1)]
        return combs

# ======================================================================
# 77. Combinations
# Topic : backtracking
# ======================================================================



# ======================================================================
# 78. Subsets
# Topic : dfs
# ======================================================================



# ======================================================================
# 79. Word Search
# Topic : dfs, backtrack
# ======================================================================


# ======================================================================
# 88. Word Search
# Topic : dfs, backtrack
# ======================================================================



# ======================================================================
# 90. Subsets II
# Topic : append
# ======================================================================

class Solution(object):
    def subsetsWithDup(self, nums):

        rst = [[]] # len(rst) = 1
        nums.sort()

        for i in range(len(nums)):
            if i == 0 or nums[i] != nums[i-1]:
                l = len(rst) # 1 -> 2
            for j in range(len(rst)-l, len(rst)): # 0,1 -> 0,2
                # 0 : rst[0]+[nums[0]] = [[], [1]]
                # 0 : rst[0]+[nums[1]] = [[], [1], [2]]
                # 1 : rst[1]+[nums[1]] = [[], [1], [2], [1,2]]
                rst.append(rst[j] + [nums[i]])
        return rst

# ======================================================================
# 92. Reverse Linked List II
# Topic : LinkedList
# ======================================================================

class Solution(object):
    def reverseBetween(self, head, left, right):

        if not head or not head.next or left == right:
            return head

        dummy = ListNode(-1, next=head)
        prev = dummy

        for _ in range(left-1): # 2-1 = 1
            prev = prev.next # 1
        
        cur = prev.next # 2

        for _ in range(right - left):
            next_ = cur.next       # 3 -> 4
            cur.next = next_.next  # 4 -> 5
            next_.next = prev.next # 2 -> 3
            prev.next = next_      # 3 -> 4

        return dummy.next

# ======================================================================
# 93. Restore IP Addresses
# Topic : dfs, backtracking
# ======================================================================

class Solution(object):
    def restoreIpAddresses(self, s):
        
        rst = []
        self.dfs(s, 0, "", rst)
        return rst

    def dfs(self, s, idx, path, rst):

        if idx > 4:
            return
        # when idx == 4, s should not have any more letters
        if idx == 4 and not s:
            rst.append(path[:-1])

        for i in range(1, len(s)+1):
            # first letter is 0 or  0 < so far letters < 256
            if s[:i] == '0' or (s[0] != '0' and 0 < int(s[:i]) < 256):
                self.dfs(s[i:], idx+1, path+s[:i]+".", rst)

# ======================================================================
# 97. Interleaving String
# Topic : dynamic programming
# ======================================================================

class Solution(object):
    def isInterleave(self, s1, s2, s3):
        m = len(s1)
        n = len(s2)

        if m+n != len(s3): return False   

        dp = [[False for j in range(n+1)] for i in range(m+1)]
        dp[0][0] = True           
        
        for i in range(0, m+1):
            for j in range(0, n+1):
                if (i>0 and dp[i-1][j] and s1[i-1] == s3[i+j-1]) or \
                (j> 0 and dp[i][j-1] and s2[j-1] == s3[i+j-1]):
                    dp[i][j] = True

        return dp[-1][-1]

# ======================================================================
# 98. Validate Binary Search Tree
# Topic : BST
# ======================================================================

class Solution(object):
    def isValidBST(self, root):

        out = []
        self.inOrder(root, out)

        for i in range(1, len(out)):
            if out[i-1] >= out[i]:
                return False
        return True


    def inOrder(self, root, out):
        if not root:
            return
        self.inOrder(root.left, out)
        out.append(root.val)
        self.inOrder(root.right, out)

# ======================================================================
# 103. Binary Tree Zigzag Level Order Traversal
# Topic : Binary Tree, while, queue
# ======================================================================

class Solution(object):
    def zigzagLevelOrder(self, root):
        queue = [(root, 0)]
        rst = []
        while queue:
            node, level = queue.pop() # 3, 0 -> 9, 1 -> 20, 1 -> 15, 2
            if node:
                if len(rst) <= level: # 0 <= 0 -> 1 <= 1 -> 2 <= 1 -> 2 <= 2
                    rst.append([]) # [[]] -> [[3], []] -> [[3], [20, 9], []]
                if level % 2 == 0:
                    # left value first
                    rst[level].append(node.val) # [[3]] -> [[3], [20, 9], [15]]
                else:
                    # right value first
                    rst[level] = [node.val] + rst[level] # [[3], [9]] -> [[3], [20, 9]]
                queue.append((node.right, level+1)) # [(20, 1)] -> [(7, 2)]
                queue.append((node.left, level+1)) # [(20, 1), (9, 1)] -> [(7, 2), (15, 2)]

        return rst

# ======================================================================
# 105. Construct Binary Tree from Preorder and Inorder Traversal
# Topic : Binary Tree, Preorder, Inorder
# ======================================================================

class Solution(object):
    def buildTree(self, preorder, inorder):
        if inorder:
            idx = inorder.index(preorder.pop(0))
            root = TreeNode(inorder[idx])
            root.left = self.buildTree(preorder, inorder[:idx]) ## left first!!!
            root.right = self.buildTree(preorder, inorder[idx+1:])

            return root

# ======================================================================
# 106. Construct Binary Tree from Inorder and Postorder Traversal
# Topic : Binary Tree, Preorder, Inorder
# ======================================================================

class Solution(object):
    def buildTree(self, inorder, postorder):
        if inorder:
            idx = inorder.index(postorder.pop())
            root = TreeNode(inorder[idx])
            root.right = self.buildTree(inorder[idx+1:], postorder) ## right first!!!
            root.left = self.buildTree(inorder[:idx], postorder)
            return root

# ======================================================================
# 116. Populating Next Right Pointers in Each Node
# Topic : Binary Tree
# ======================================================================

class Solution(object):
    def connect(self, root):

        if not root:
            return root

        if root.left:
            left, right = root.left, root.right
            self.connect(left) 
            self.connect(right)

            while left:
                left.next = right
                left, right = left.right, right.left
        return root

# ======================================================================
# 122. Best Time to Buy and Sell Stock II
# Topic : for
# ======================================================================

class Solution(object):
    def maxProfit(self, prices):
        profit = 0
        start = prices[0]

        for i in range(1, len(prices)):
            if start < prices[i]:
                profit += prices[i] - start

            start = prices[i]
        return profit

# ======================================================================
# 128. Longest Consecutive Sequence
# Topic : sort, set
# ======================================================================

class Solution(object):
    def longestConsecutive(self, nums):

        if not nums:
            return 0
        
        nums = sorted(set(nums))

        rst, cur = 1, 1
        for i in range(1, len(nums)):
            if nums[i-1] == nums[i] - 1:
                cur += 1
                rst = max(rst, cur)
            else:
                cur = 1

        return rst

# ======================================================================
# 134. Gas Station
# Topic : greedy method
# Greedy Method
# - Goal is to find the best solution from a set of feasible solutions.
# - It is a strategy for solving optimization problems.
# ======================================================================

class Solution(object):
    def canCompleteCircuit(self, gas, cost):

        tank, cur, start = 0, 0, 0
        for i in range(len(gas)):
            tank += gas[i] - cost[i]
            cur += gas[i] - cost[i]

            if cur < 0:
                start = i+1
                cur = 0

        return start if tank >= 0 else -1

# ======================================================================
# 138. Copy List with Random Pointer
# Topic : Linked List
# ======================================================================

# ======================================================================
# 139. Word Break
# Topic : string
# ======================================================================

class Solution(object):
    def wordBreak(self, s, wordDict):

        ok = [True]
        for i in range(1, len(s)+1): # (1,9)
            # if both conditions are true for any j -> True
            ok += [any(ok[j] and s[j:i] in wordDict for j in range(i))]
            # i=1 [True, False]
            # i=2 [True, False, False]
            # i=3 [True, False, False, False]
            # i=4 [True, False, False, False, True] ###
            # i=5 [True, False, False, False, True, False]
            # i=6 [True, False, False, False, True, False, False]
            # i=7 [True, False, False, False, True, False, False, False]
            # i=8 [True, False, False, False, True, False, False, False, True] ###

        return ok[-1]

# ======================================================================
# 143. Reorder List
# Topic : Linked List, reverse, slow, fast, mid
# ======================================================================

class Solution(object):
    def reorderList(self, head):
        
        if head is None:
            return

        slow = head # 1
        fast = head.next # 2

        # Catch middle
        while fast and fast.next:
            fast = fast.next.next # last : 4
            slow = slow.next # middle : 2

        # Reverse the latter half
        rev = None
        cur = slow.next # 3
        while cur: # 3 -> 4
            rev, rev.next, cur = cur, rev, cur.next # 3-None, 4 -> 4-3-None, None 

        slow.next = None

        # Connect
        while rev: # 4-3-None
            h_next = head.next # 2
            r_next = rev.next # 3
            head.next = rev   # 4
            rev.next = h_next # 2
            rev = r_next      # 3
            head = h_next      # 2

# ======================================================================
# 146. LRU Cache
# Topic : HashMap, put, get
# ======================================================================          

class LRUCache(object):

    def __init__(self, capacity):
        
        # ex) 2 : it can hold at most 2 key-pair values at a time
        self.capacity = capacity
        # retain the original insertion order of items
        self.cache = OrderedDict()

    def get(self, key):
        
        if key not in self.cache:
            return -1
        
        # if the key accessed, the position should be moved the last
        self.cache[key] = self.cache.pop(key)
        return self.cache[key]

    def put(self, key, value):
       
        if key in self.cache:
            self.cache.pop(key)

        # if it's full
        elif len(self.cache) == self.capacity:
            # popitem : remove and return a (key, val) pair
            # last=False : remove the first item
            self.cache.popitem(last=False)

        self.cache[key] = value

# ======================================================================
# 153. Find Minimum in Rotated Sorted Array
# Topic : Array, mid
# ======================================================================          

class Solution(object):
    def findMin(self, nums):

        left = 0
        right = len(nums) - 1

        while left < right:
            mid = (left+right) // 2
            if nums[right] < nums[mid]:
                left = mid+1
            else:
                right = mid

        return nums[left]

# ======================================================================
# 186. Reverse Words in a String II
# Topic : string, in-place reverse
# ======================================================================          

class Solution(object):
    def reverseWords(self, s):

        def reverse(left, right):
            while left < right:
                s[left], s[right] = s[right], s[left]
                left += 1
                right -= 1

        reverse(0, len(s)-1) # reverse all

        left = 0
        for idx, ch in enumerate(s):
            if ch == " ":
                reverse(left, idx-1) # eulb -> blue
                left = idx +1 # next to " "
        
        # last word
        reverse(left, len(s)-1)
        
# ======================================================================
# 198. House Robber
# Topic : swap, adjacent
# ======================================================================    

class Solution(object):
    def rob(self, nums):
        
        bf3, bf2, adj = 0, 0, 0
        for cur in nums:
            bf3, bf2, adj = bf2, adj, cur + max(bf3, bf2)

        return max(bf2, adj)

# ======================================================================
# 200. Number of Islands
# Topic : dfs
# ======================================================================    

class Solution(object):
    def numIslands(self, grid):
        
        if not grid: return 0

        rows = len(grid)
        cols = len(grid[0])

        def dfs(row, col):
            if row < 0 or col < 0 or row >= rows or col >= cols or grid[row][col] != '1':
                return
            
            grid[row][col] = '0'
            dfs(row-1, col)
            dfs(row+1, col)
            dfs(row, col-1)
            dfs(row, col+1)

        cnt = 0
        for row in range(rows):
            for col in range(cols):
                if grid[row][col] == '1':
                    dfs(row, col)
                    cnt += 1

        return cnt

# ======================================================================
# 207. Course Schedule
# Topic : dfs
# ======================================================================    

class Solution(object):
    def canFinish(self, numCourses, prerequisites):

        graph = [[] for _ in range(numCourses)] # [[], []]
        visit = [0 for _ in range(numCourses)] # [0, 0]

        # create graph
        for x, y in prerequisites:
            graph[x].append(y)

        # visit each node
        for i in range(numCourses):
            if not self.dfs(i, graph, visit):
                return False
        
        return True

    def dfs(self, i, graph, visit):

        # if the node is marked as being visited
        if visit[i] == -1:
            return False

        # if it's done visited
        if visit[i] == 1:
            return True

        # mark as visit
        visit[i] = -1

        # visit all the neighbors
        for j in graph[i]:
            if not self.dfs(j, graph, visit):
                return False
        
        # after visit all the neighbors, mark it as done
        visit[i] = 1
        return True

# ======================================================================
# 208. Implement Trie (Prefix Tree)
# Topic : dic
# ======================================================================    

class Trie(object):

    def __init__(self):
        self.trie = {}

    def insert(self, word):

        t = self.trie

        for c in word:
            if c not in t:
                t[c] = {}
            t = t[c] # {'a':{}}
        t["-"] = True
        # {'a': {'p': {'p': {'l': {'e': {'-': True}}}}}}
        # self.trie['a']['p']

    def search(self, word):

        t = self.trie
        for c in word:
            if c not in t: return False
            t = t[c]
        return "-" in t        

    def startsWith(self, prefix):

        t = self.trie
        for c in prefix:
            if c not in t:return False
            t = t[c]
        return True

# ======================================================================
# 210. Course Schedule II
# Topic : dfs
# ======================================================================    

# ======================================================================
# 211. Design Add and Search Words Data Structure
# Topic : dfs
# ======================================================================    


# ======================================================================
# 213. House Robber II
# Topic : swap, adjacent
# ======================================================================    

class Solution(object):
    def rob(self, nums):
        
        def simple(nums):
            bf3, bf2, adj = 0, 0, 0

            for cur in nums:
                bf3, bf2, adj = bf2, adj, max(cur+bf2, adj)
            return max(bf2, adj)

        if len(nums) <= 1:
            return sum(nums)

        return max(simple(nums[1:]), simple(nums[:len(nums)-1]))

# ======================================================================
# 215. Kth Largest Element in an Array
# Topic : heap, heapq, heappush, heappop
# num.sort(reverse = True)
# ======================================================================    

class Solution(object):
    def findKthLargest(self, nums, k):

        heap = []
        for num in nums:
             # heap :
             # [3]
             # -> [2, 3] : since 2 is small, it becomes root
             # -> [1, 3, 2] : since 1 is small, it becomes root and two become children
             # -> [2, 3, 5] -> [3, 5, 6] -> [4, 6, 5]
            heapq.heappush(heap, num)
            if len(heap) > k:
                heapq.heappop(heap) # remove the smallest element
                # [2, 3] -> [3, 5] -> [5, 6] -> [5, 6]
        
        return heap[0]
        
# ======================================================================
# 227. Basic Calculator II
# Topic : heap, heapq, heappush, heappop
# ======================================================================    

class Solution(object):
    def calculate(self, s):
        total = 0
        # ['+', 'a', '+', 'b', '-', 'c']
        outer = iter(['+'] + re.split('([+-])', s))

        for addsub in outer:
            inner = iter(['*'] + re.split('([*/])', next(outer)))
            term = 1
            for muldiv in inner:
                n = int(next(inner))
                term = term * n if muldiv == "*" else term/n
            total += term if addsub == '+' else -term

        return total

# ======================================================================
# 236. Lowest Common Ancestor of a Binary Tree
# Topic : binary tree
# ======================================================================    

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):

        if not root or root == p or root == q:
            return root

        # find p, q
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)

        if left and right:
            return root

        return left if left else right
 
# ======================================================================
# 238. Product of Array Except Self
# Topic : Array
# ======================================================================    

class Solution(object):
    def productExceptSelf(self, nums):

        n = len(nums) # [1, 2, 3, 4]
        cal = [1] * n # [1, 1, 1, 1]

        for i in range(1, n):
            cal[i] = cal[i-1] * nums[i-1] # [1, 1, 2, 6]

        right = nums[-1] # 4
        for i in range(n-2, -1, -1):
            cal[i] *= right
            # [1,1,8,6] -> [1,12,8,6] -> [24,12,8,6]
            right *= nums[i] # 12 -> 24
            
        return cal

# ======================================================================
# 253. Meeting Rooms II
# Topic : Array, heap, heapq, heapreplace, heappush
# ======================================================================    

class Solution(object):
    def minMeetingRooms(self, intervals):
        
        intervals.sort(key = lambda x: x[0])
        heap = []

        for arr in intervals:
            if heap and arr[0] >= heap[0]:
                heapq.heapreplace(heap, arr[1])
            else:
                heapq.heappush(heap, arr[1])
            # print(heap) # [30], [10, 30], [20, 30]
        return len(heap)

# ======================================================================
# 285. Inorder Successor in BST
# Topic : BST, while
# ======================================================================    

class Solution(object):
    def inorderSuccessor(self, root, p):

        suc = None
        while root:
            if p.val < root.val:
                suc = root
                root = root.left
            else:
                root = root.right
        return suc

# ======================================================================
# 287. Find the Duplicate Number
# Topic : dic
# ======================================================================    

class Solution(object):
    def findDuplicate(self, nums):

        dic = {}
        for n in nums:
            if n not in dic:
                dic[n] = 0
            else:
                return n
        
# ======================================================================
# 300. Longest Increasing Subsequence
# Topic : while
# ======================================================================    

class Solution(object):
    def lengthOfLIS(self, nums):

        tmp = [0 for _ in range(len(nums))]
        size = 0

        for x in nums:
            i, j = 0, size
            while i != j:
                m = (i + j) // 2
                if tmp[m] < x:
                    i = m+1
                else:
                    j = m
            tmp[i] = x
            size = max(i+1, size)

        # tmp : [10, 0, 0, 0, 0, 0, 0, 0], size : 1
        # tmp : [9, 0, 0, 0, 0, 0, 0, 0], size : 1
        # tmp : [2, 0, 0, 0, 0, 0, 0, 0], size : 1
        # tmp : [2, 5, 0, 0, 0, 0, 0, 0], size : 2
        # tmp : [2, 3, 0, 0, 0, 0, 0, 0], size : 2
        # tmp : [2, 3, 7, 0, 0, 0, 0, 0], size : 3
        # tmp : [2, 3, 7, 101, 0, 0, 0, 0], size : 4
        # tmp : [2, 3, 7, 18, 0, 0, 0, 0], size : 4

        return size

# ======================================================================
# 316. Remove Duplicate Letters
# Topic : while
# ======================================================================    

class Solution(object):
    def removeDuplicateLetters(self, s):

        idx = {c: i for i, c in enumerate(s)}
        rst = ''

        for i, c in enumerate(s):
            if c not in rst:
                # rst[-1:] 로 안 하면 out of index 문제 발생 (=rst[-1])
                while c < rst[-1:] and i < idx[rst[-1]]:
                    rst = rst[:-1]
                rst += c

        return rst

# ======================================================================
# 322. Coin Change
# Topic : BFS
# ======================================================================    

class Solution(object):
    def coinChange(self, coins, amount):

        if amount == 0:
            return 0

        q = deque()
        q.append(amount)
        visited = set()
        depth = 0

        while q:
            for i in range(len(q)):
                amt = q.popleft()

                if amt < 0: continue
                elif amt == 0: return depth

                if amt not in visited:
                    visited.add(amt)

                    for c in coins:
                        q.append(amt - c)
            
            depth += 1

        return -1

# ======================================================================
# 328. Odd Even Linked List
# Topic : Linked List
# ======================================================================    

class Solution(object):
    def oddEvenList(self, head):

        dummy1 = odd = ListNode(0)
        dummy2 = even = ListNode(0)

        while head:
            odd.next = head
            even.next = head.next
            odd = odd.next
            even = even.next

            head = head.next.next if even else None

        odd.next = dummy2.next
        return dummy1.next

# ======================================================================
# 347. Top K Frequent Elements
# Topic : dic
# dic sort with values : sorted(freq.items(), key=lambda x: x[1], reverse=True)
# ======================================================================   

class Solution(object):
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        
        freq = {}
        for n in nums:
            if n in freq:
                freq[n] += 1
            else:
                freq[n] = 1

        # (1, 3), (2, 2), (3, 1)
        freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)

        rst = []
        for i in range(k):
            rst.append(freq[i][0])
        return rst


# ======================================================================
# 380. Insert Delete GetRandom O(1)
# Topic : 
# ======================================================================

# ======================================================================
# 399. Evaluate Division
# Topic : BFS
# ======================================================================














