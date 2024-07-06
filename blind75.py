# https://leetcode.com/discuss/general-discussion/460599/blind-75-leetcode-questions
# https://hackernoon.com/14-patterns-to-ace-any-coding-interview-question-c5bb3357f6ed

# ========================================================
# Array : 10 questions
# ========================================================

# 1. Two Sum (Easy)

class Solution(object):
    def twoSum(self, nums, target):

        rst = {}
        for i, n in enumerate(nums):
            diff = target - n
            if diff in rst:
                return [rst[diff], i]
            rst[n] = i

        return []

# 121. Best Time to Buy and Sell Stock (Easy)

class Solution(object):
    def maxProfit(self, prices):

        min_pr = prices[0]
        max_pf = 0

        for p in prices[1:]:
            max_pf = max(max_pf, p - min_pr)
            min_pr = min(min_pr, p)

        return max_pf

# 217. Contains Duplicate (Easy)

class Solution(object):
    def containsDuplicate(self, nums):
        return len(set(nums)) != len(nums)

# 238. Product of Array Except Self (Medium)

class Solution(object):
    def productExceptSelf(self, nums):

        n = len(nums)
        rst = [1] * n

        for i in range(1, n):
            rst[i] *= nums[i-1] * rst[i-1]

        right = nums[-1]
        for i in range(n-2, -1, -1):
            rst[i] *= right
            right *= nums[i]

        return rst

# 53. Maximum Subarray (Medium)

class Solution(object):
    def maxSubArray(self, nums):

        max_v = nums[0]
        arr = [max_v]

        for i in range(1, len(nums)):
            cur = max(arr[i-1] + nums[i], nums[i])
            arr.append(cur)

            if cur > max_v:
                max_v = cur

        return max_v

# 152. Maximum Product Subarray (Medium)

class Solution(object):
    def maxProduct(self, nums):

        max_v = nums[0]
        min_v = nums[0]
        rst = nums[0]

        for n in nums[1:]:
            max_v *= n
            min_v *= n
            max_v, min_v = max(max_v, min_v, n), min(max_v, min_v, n)
            rst = max(rst, max_v)
        
        return rst

# 153. Find Minimum in Rotated Sorted Array (Medium)

class Solution(object):
    def findMin(self, nums):

        left = 0
        right = len(nums)-1
        
        while left < right:
            mid = (left + right) // 2
            if nums[right] < nums[mid]:
                left = mid + 1
            else:
                right = mid

        return nums[left]

# 33. Search in Rotated Sorted Array (Medium)

class Solution(object):
    def search(self, nums, target):
        
        left = 0
        right = len(nums) - 1

        while left <= right:
            mid = (left + right) //2
            if nums[mid] == target:
                return mid
            elif nums[left] <= nums[mid]:
                if nums[left] <= target < nums[mid]:
                    right = mid -1
                else:
                    left = mid +1
            else:
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1

        return -1

# 15. 3Sum (Medium)

class Solution(object):
    def threeSum(self, nums):

        rst = set()

        n, p, z = [], [], []
        for num in nums:
            if num == 0:
                z.append(num)
            elif num < 0:
                n.append(num)
            else:
                p.append(num)
        
        N, P = set(n), set(p)

        if len(z) >= 3:
            rst.add((0,0,0))

        if z:
            for i in P:
                if -i in N:
                    rst.add((-i, 0, i))
        
        for i in range(len(n)):
            for j in range(i+1, len(n)):
                target = - (n[i] + n[j])
                if target in P:
                    rst.add(tuple(sorted([n[i], n[j], target])))

        for i in range(len(p)):
            for j in range(i+1, len(p)):
                target = - (p[i] + p[j])
                if target in N:
                    rst.add(tuple(sorted([target, p[i], p[j]])))

        return list(rst)

# 11. Container With Most Water (Medium)

class Solution(object):
    def maxArea(self, height):

        rst = 0
        i, j = 0, len(height) - 1

        while i < j:
            w = j - i
            if height[i] < height[j]:
                rst = max(rst, w * height[i])
                i += 1
            else:
                rst = max(rst, w * height[j])
                j -= 1
        return rst

# ========================================================
# Binary : 5 questions
# ========================================================

# 371. Sum of Two Integers (Medium) ##

# 191. Number of 1 Bits (Easy) ##

class Solution(object):
    def hammingWeight(self, n):

        # n & (n - 1): a bitwise AND operation
        # 6 : 110
        # 5 : 101
        # --------
        #     100

        cnt = 0
        while n != 0:
            n &= (n-1)
            cnt += 1
        return cnt
        
# 338. Counting Bits (Easy) ##

# 268. Missing Number (Easy)

class Solution(object):
    def missingNumber(self, nums):

        sums = sum(range(len(nums)+1))
        return sums - sum(nums)

# 190. Reverse Bits (Easy) ##

# ========================================================
# Dynamic Programming : 5 questions
# ========================================================

# 70. Climbing Stairs (Easy)

class Solution(object):
    def climbStairs(self, n):

        if n <= 3:
            return n

        dp = [0,1,2]

        for i in range(3, n+1):
            dp.append(dp[i-2]+dp[i-1])

        return dp[-1]

# 322. Coin Change (Medium)

class Solution(object):
    def coinChange(self, coins, amount):

        max_val = amount + 1 # greater than max possible number of coins
        dp = [max_val] * (amount + 1) # dp: min number of coins needed

        dp[0] = 0 # no coins are needed to make 0

        for coin in coins:
            for x in range(coin, amount+1):
                # min coins needed without considering the current coin
                # min coins needed for the remaining amount + current coin
                dp[x] = min(dp[x], dp[x-coin]+1)
        
        return dp[amount] if dp[amount] != max_val else -1

# 300. Longest Increasing Subsequence (Medium)

class Solution(object):
    def lengthOfLIS(self, nums):

        dp = [0] * len(nums)
        size = 0

        for n in nums:
            i, j = 0, size
            while i != j:
                m = (i+j) / 2
                if dp[m] < n:
                    i = m+1
                else:
                    j = m
            dp[i] = n
            size = max(size, i+1)
        
        return size

# 1143. Longest Common Subsequence (Medium)

class Solution(object):
    def longestCommonSubsequence(self, text1, text2):
        
        m = len(text1) + 1
        n = len(text2) + 1
        dp = [[0]*n for _ in range(m)]

        # [[0, 0, 0, 0],
        #  [0, 1, 1, 1], # a -> a,c,e
        #  [0, 1, 1, 1], # b -> a,c,e
        #  [0, 1, 2, 2], # c -> a,c,e
        #  [0, 1, 2, 2], # d -> a,c,e
        #  [0, 1, 2, 3]] # e -> a,c,e

        for i in range(1, m):
            for j in range(1, n):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = 1 + dp[i-1][j-1]
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[-1][-1]

# 139. Word Break (Medium)

class Solution(object):
    def wordBreak(self, s, wordDict):

        dp = [True]
        for i in range(1, len(s)+1):
            dp += [any(dp[j] and s[j:i] in wordDict for j in range(i))]
            
        # [True, False, False, False, True, False, False, False, True]
        return dp[-1]

# 39. Combination Sum (Medium)

class Solution(object):
    def combinationSum(self, candidates, target):

        rst = []
        candidates.sort()

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

# 198. House Robber (Medium)

class Solution(object):
    def rob(self, nums):

        bf3, bf2, adj = 0, 0, 0
        for cur in nums:
            bf3, bf2, adj = bf2, adj, max(bf2 + n, adj)
        
        return max(bf2, adj)

# 213. House Robber II (Medium)

class Solution(object):
    def rob(self, nums):

        def simple(nums):
            bf3, bf2, adj = 0, 0, 0

            for cur in nums:
                bf3, bf2, adj = bf2, adj, max(bf2+cur, adj)
            return max(bf2, adj)

        if len(nums) <= 1:
            return sum(nums)

        return max(simple(nums[1:]), simple(nums[:len(nums)-1]))

# 91. Decode Ways (Medium)

class Solution(object):
    def numDecodings(self, s):

        if not s: return 0

        dp = [0] * (len(s)+1)
        dp[0] = 1
        dp[1] = 0 if s[0] == '0' else 1

        for i in range(2, len(s)+1):
            if 0 < int(s[i-1:i]) <= 9:
                dp[i] += dp[i-1]
            
            if 10 <= int(s[i-2:i]) <= 26:
                dp[i] += dp[i-2]

        return dp[len(s)]

# 62. Unique Paths (Medium)

class Solution(object):
    def uniquePaths(self, m, n):
        
        dp = [[1 for _ in range(n)] for _ in range(m)]

        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]

        return dp[-1][-1]

# 55. Jump Game (Medium)

class Solution(object):
    def canJump(self, nums):
        
        last = len(nums)-1

        for i in range(len(nums)-2, -1, -1):
            if i + nums[i] >= last:
                last = i
        return last == 0

# ========================================================
# Graph : 8 questions
# ========================================================

# 133. Clone Graph (Medium) ##

# 207. Course Schedule (Medium) ##

# 417. Pacific Atlantic Water Flow (Medium) ##

# 200. Number of Islands (Medium)

class Solution(object):
    def numIslands(self, grid):
        
        if not grid: return 0

        rows, cols = len(grid), len(grid[0])

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

# 128. Longest Consecutive Sequence (Medium)

class Solution(object):
    def longestConsecutive(self, nums):

        if not nums: return 0

        nums = sorted(set(nums))
        cnt = 1
        rst = 1

        for i in range(1, len(nums)):
            if nums[i-1] == nums[i] -1:
                cnt += 1
                rst = max(rst, cnt)
            else:
                cnt = 1

        return rst

# 269. Alien Dictionary (Hard) ##

# 261. Graph Valid Tree (Medium)

class Solution(object):
    def validTree(self, n, edges):
        # valid tree : no loop, edge = n-1

        if len(edges) != n-1:
            return False

        # edges = [[0,1],[0,2],[0,3],[1,4]]
        # neighbors : {0: [], 1: [], 2: [], 3: [], 4: []}
        neighbors = {i : [] for i in range(n)}
        for v, w in edges:
            neighbors[v] += w,
            neighbors[w] += v,
            # {0: [1, 2, 3], 1: [0, 4], 2: [0], 3: [0], 4: [1]}

        def visit(v):
            # remove the visited nodes from the neighbors
            # if there's no v, return []
            map(visit, neighbors.pop(v, []))

        visit(0)
        return not neighbors

# 323. Number of Connected Components in an Undirected Graph (Medium)

class Solution(object):
    def countComponents(self, n, edges):
        
        graph = defaultdict(list)
        
        for x, y in edges:
            graph[x].append(y)
            graph[y].append(x)

        def dfs(node, seen):
            seen.add(node)
            for neighbor in graph[node]:
                if neighbor not in seen:
                    dfs(neighbor, seen)

        cnt = 0
        seen = set()
        for node in range(n):
            if node not in seen:
                dfs(node, seen)
                cnt += 1
        
        return cnt

# ========================================================
# Interval : 5 questions
# ========================================================

# 57. Insert Interval (Medium)

class Solution(object):
    def insert(self, intervals, newInterval):

        s, e = newInterval[0], newInterval[1]
        left, right = [], []

        for lst in intervals:
            if lst[1] < s:
                left += lst, ### comma is needed!
            elif lst[0] > e:
                right += lst,
            else:
                s = min(s, lst[0])
                e = max(e, lst[1])
        return left + [[s, e]] + right

# 56. Merge Intervals (Medium)

class Solution(object):
    def merge(self, intervals):

        intervals.sort(key=lambda x: x[0])

        rst = []
        for lst in intervals:

            if rst and lst[0] <= rst[-1][1]:
                rst[-1][1] = max(rst[-1][1], lst[1])
            else:
                rst += lst,

        return rst

# 435. Non-overlapping Intervals (Medium)

class Solution(object):
    def eraseOverlapIntervals(self, intervals):

        intervals.sort()
        cnt = 0
        prev = float("-inf")

        for lst in intervals:
            if lst[0] >= prev:
                prev = lst[1]
            else:
                cnt += 1
                prev = min(prev, lst[1])
        return cnt        

# 252. Meeting Rooms (Easy)

class Solution(object):
    def canAttendMeetings(self, intervals):

        intervals.sort()
        for i, lst in enumerate(intervals):
            if i != 0 and lst[0] < intervals[i-1][1]:
                return False

        return True

# 253. Meeting Rooms II (Medium)

class Solution(object):
    def minMeetingRooms(self, intervals):

        intervals.sort() # [[0,30],[5,10],[15,20]]
        rst = []

        for lst in intervals:
            # the start time of the current meeting >= end time of the earliest ending meeting
            if rst and lst[0] >= rst[0]:
                heapq.heapreplace(rst, lst[1])
            # if the current meeting starts before the earliest ending meeting ends
            else:
                heapq.heappush(rst, lst[1])
            # print(rst) : [30] -> [10, 30] -> [20, 30]

        return len(rst)

# ========================================================
# Linked List : 6 questions
# ========================================================

# 206. Reverse Linked List (Easy)

class Solution(object):
    def reverseList(self, head):

        prev = None
        while head:
            next_p = head.next
            head.next = prev
            prev = head
            head = next_p

        return prev

# 141. Linked List Cycle (Easy)

class Solution(object):
    def hasCycle(self, head):

        try:
            s = head
            e = head.next
            while s is not e:
                s = s.next
                e = e.next.next
            return True
        # If there is no cycle,
        # the fast pointer will reach the end of the list,
        # causing an exception
        except:
            return False

# 21. Merge Two Sorted Lists (Easy)

class Solution(object):
    def mergeTwoLists(self, list1, list2):
        
        head = dummy = ListNode(-1)

        while list1 and list2:
            if list1.val <= list2.val:
                dummy.next = list1
                list1 = list1.next
            else:
                dummy.next = list2
                list2 = list2.next
            dummy = dummy.next

        dummy.next = list1 if list1 else list2

        return head.next

# 23. Merge K Sorted Lists (Hard) ##

# 19. Remove Nth Node From End Of List (Medium)

class Solution(object):
    def removeNthFromEnd(self, head, n):

        fast, slow = head, head

        for _ in range(n):
            fast = fast.next

        if not fast:
            return head.next
        
        while fast.next:
            slow = slow.next
            fast = fast.next
        
        slow.next = slow.next.next
        return head

# 143. Reorder List (Medium)

class Solution(object):
    def reorderList(self, head):

        if not head or not head.next:
            return

        mid = self.midNode(head) # ex1) 2, ex2) 3
        half2 = mid.next # ex1) 3->4, ex2) 4->5
        mid.next = None # ex1) 2->None, ex2) 3->None

        half2 = self.reverseNode(half2) # ex1) 4->3, ex2) 5->4

        c1, c2 = head, half2 # 1->2 ... , 4->3
        f1, f2 = None, None

        while c1 and c2:
            # Backup
            f1 = c1.next
            f2 = c2.next

            # Link
            c1.next = c2
            c2.next = f1

            # Move
            c1 = f1
            c2 = f2

    def midNode(self, head):
        fast, slow = head, head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        
        return slow

    def reverseNode(self, head):

        prev = None

        while head:
            next_p = head.next
            head.next = prev
            prev = head
            head = next_p

        return prev

# ========================================================
# Matrix : 4 questions
# ========================================================

# 73. Set Matrix Zeroes (Medium)

class Solution(object):
    def setZeroes(self, matrix):

        if not matrix:
            return []

        m, n = len(matrix), len(matrix[0])

        z_rows = [False] * m
        z_cols = [False] * n

        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 0:
                    z_rows[i] = True
                    z_cols[j] = True

        for i in range(m):
            for j in range(n):
                if z_rows[i] or z_cols[j]:
                    matrix[i][j] = 0 

# 54. Spiral Matrix (Medium)

class Solution(object):
    def spiralOrder(self, matrix):

        rst = []
        while matrix:

            rst += matrix.pop(0)

            if matrix and matrix[0]:
                for row in matrix:
                    rst.append(row.pop())

            if matrix:
                rst += matrix.pop()[::-1]
            
            if matrix and matrix[0]:
                for row in matrix[::-1]:
                    rst.append(row.pop(0))

        return rst

# 48. Rotate Image (Medium)

class Solution(object):
    def rotate(self, matrix):

        matrix.reverse()

        for i in range(len(matrix)):
            for j in range(i):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

        return matrix

# 79. Word Search (Medium) ##

# ========================================================
# String : 10 questions
# ========================================================

# 3. Longest Substring Without Repeating Characters (Medium)

class Solution(object):
    def lengthOfLongestSubstring(self, s):

        len_s = 0
        idx = {}

        i = 0
        for j in range(len(s)):
            if s[j] in idx:
                i = max(i, idx[s[j]])
            len_s = max(len_s, j-i+1)
            idx[s[j]] = j+1

        return len_s

# 424. Longest Repeating Character Replacement (Medium) ##

# 76. Minimum Window Substring (Hard) ##

# 242. Valid Anagram (Easy)

class Solution(object):
    def isAnagram(self, s, t):

        if len(s) != len(t):
            return False

        for char in set(s):
            if s.count(char) != t.count(char):
                return False
        return True

# 49. Group Anagrams (Medium)

class Solution(object):
    def groupAnagrams(self, strs):

        rst = []
        dic = {}

        for s in strs:
            word = ''.join(sorted(s))
            if word in dic:
                rst[dic[word]].append(s)
            else:
                dic[word] = len(rst)
                rst.append([s])
        
        return rst

# 20. Valid Parentheses (Easy)

class Solution(object):
    def isValid(self, s):

        stk = []
        for w in s:
            if w in '({[':
                stk.append(w)
            else:
                if not stk \
                or w == ')' and stk[-1] != '(' \
                or w == ']' and stk[-1] != '[' \
                or w == '}' and stk[-1] != '{':
                    return False
                stk.pop()

        return not stk

# 125. Valid Palindrome (Easy)

class Solution(object):
    def isPalindrome(self, s):

        # i, j = 0, len(s)-1

        # while i < j:
        #     while i < j and not s[i].isalnum():
        #         i += 1
        #     while i < j and not s[j].isalnum():
        #         j -= 1
            
        #     if s[i].lower() != s[j].lower():
        #         return False
            
        #     i += 1
        #     j -= 1

        # return True

        s = s.lower()
        s = re.sub('[^0-9a-z]', "", s)
        s = s.replace(" ", "")
        return s == s[::-1]

# 5. Longest Palindromic Substring (Medium)

class Solution(object):
    def longestPalindrome(self, s):

        if s == s[::-1]:
            return s

        start, end = 1, 0
        for i in range(1, len(s)):
            left, right = i-end, i+1
            s1, s2 = s[left-1:right], s[left:right]
            if left-1 >= 0 and s1 == s1[::-1]:
                start, end = left-1, end+2
            elif s2 == s2[::-1]:
                start, end = left, end+1
        
        return s[start:start+end]

# 647. Palindromic Substrings (Medium) ##

# 271. Encode and Decode Strings (Medium) ##

# ========================================================
# Tree : 14 questions
# ========================================================

# 104. Maximum Depth of Binary Tree (Easy)

class Solution(object):
    def maxDepth(self, root):

        if not root:
            return 0

        left = self.maxDepth(root.left)
        right = self.maxDepth(root.right)

        return max(left, right) + 1

# 100. Same Tree (Easy)

class Solution(object):
    def isSameTree(self, p, q):

        if p and q:
            return p.val == q.val and \
            self.isSameTree(p.left, q.left) and \
            self.isSameTree(p.right, q.right)
        
        # Check if Both Nodes are None
        return p is q

# 226. Invert/Flip Binary Tree (Easy)

class Solution(object):
    def invertTree(self, root):

        if not root:
            return

        self.invertTree(root.left)
        self.invertTree(root.right)

        root.left, root.right = root.right, root.left
        return root

# 124. Binary Tree Maximum Path Sum (Hard) ##

# 102. Binary Tree Level Order Traversal (Medium) ##

class Solution(object):
    def levelOrder(self, root):

        ans, level = [], [root]
        while root and level:
            ans.append([node.val for node in level])
            pair = [(node.left, node.right) for node in level]
            level = [n for node in pair for n in node if n]
        return ans

# [n for node in pair for n in node if n]
# for node in pair:   # (9, 20)
#     for n in node:  # 9 -> 20
#         if n:
#             level.append(n)

# 297. Serialize and Deserialize Binary Tree (Hard) ##

# 572. Subtree of Another Tree (Easy)

class Solution(object):
    def isSubtree(self, root, subRoot):

        if not root:
            return False

        if self.isSameTree(root, subRoot):
            return True
        
        return self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)

    def isSameTree(self, p, q):
        if p and q:
            return p.val == q.val and self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
        return p is q
        
# 105. Construct Binary Tree from Preorder and Inorder Traversal (Medium)

class Solution(object):
    def buildTree(self, preorder, inorder):

        if inorder:
            idx = inorder.index(preorder.pop(0))
            root = TreeNode(inorder[idx])
            root.left = self.buildTree(preorder, inorder[:idx])
            root.right = self.buildTree(preorder, inorder[idx+1:])

            return root

# 98. Validate Binary Search Tree (Medium)

class Solution(object):
    def isValidBST(self, root):

        rst = []
        self.inOrder(root, rst)

        for i in range(1, len(rst)):
            if rst[i-1] >= rst[i]:
                return False
        return True
    
    def inOrder(self, root, rst):
        if not root:
            return
        
        self.inOrder(root.left, rst)
        rst.append(root.val)
        self.inOrder(root.right, rst)

# 230. Kth Smallest Element in a BST (Medium)

class Solution(object):
    def kthSmallest(self, root, k):

        rst = []
        self.inOrder(root, rst)
        return rst[k-1]

    def inOrder(self, root, rst):
        if not root:
            return

        self.inOrder(root.left, rst)
        rst.append(root.val)
        self.inOrder(root.right, rst)
        
# 235. Lowest Common Ancestor of BST (Medium)

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        
        while True:
            if root.val > p.val and root.val > q.val:
                root = root.left
            elif root.val < p.val and root.val < q.val:
                root = root.right
            else:
                return root

# 208. Implement Trie (Prefix Tree) (Medium) ##

# 211. Add and Search Word (Medium) ##

# 212. Word Search II (Hard) ##

# ========================================================
# Heap : 3 questions
# ========================================================

# 23. Merge K Sorted Lists (Hard) ##

# 347. Top K Frequent Elements (Medium)

class Solution(object):
    def topKFrequent(self, nums, k):

        freq = defaultdict(lambda:0) # collections
        for n in nums:
            freq[n] += 1

        freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:k]

        return [n[0] for n in freq]

# 295. Find Median from Data Stream (Hard) ##

