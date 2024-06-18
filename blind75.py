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
    def maxSubArray(self, nums):

        max_v = nums[0]
        arr = [max_v]

        for i in range(1, len(nums)):
            cur = max(arr[i-1] + nums[i], nums[i])
            arr.append(cur)

            if cur > max_v:
                max_v = cur

        return max_v

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
        
# 338. Counting Bits ##

# 268. Missing Number (Easy)

class Solution(object):
    def missingNumber(self, nums):

        sums = sum(range(len(nums)+1))
        return sums - sum(nums)

# 190. Reverse Bits ##

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

# 39. Combination Sum

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

# 198. House Robber

class Solution(object):
    def rob(self, nums):

        bf3, bf2, adj = 0, 0, 0
        for cur in nums:
            bf3, bf2, adj = bf2, adj, cur + max(bf3, bf2)
        
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

# 91. Decode Ways

class Solution(object):
    def numDecodings(self, s):

        if not s:
            return 0
        
        dp = [0 for _ in range(len(s)+1)]

        dp[0] = 1
        dp[1] = 0 if s[0] == '0' else 1

        for i in range(2, len(s)+1):
            if 0 < int(s[i-1:i]) <= 9:
                dp[i] += dp[i-1]
            if 10 <= int(s[i-2:i]) <= 26:
                dp[i] += dp[i-2]
        
        return dp[len(s)]

# 62. Unique Paths

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

# 133. Clone Graph (Medium)

# Course Schedule
# Pacific Atlantic Water Flow
# Number of Islands
# Longest Consecutive Sequence
# Alien Dictionary (Leetcode Premium)
# Graph Valid Tree (Leetcode Premium)
# Number of Connected Components in an Undirected Graph (Leetcode Premium)

# ========================================================
# Interval : 5 questions
# ========================================================

Insert Interval
Merge Intervals
Non-overlapping Intervals
Meeting Rooms (Leetcode Premium)
Meeting Rooms II (Leetcode Premium)

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

Set Matrix Zeroes
Spiral Matrix
Rotate Image
Word Search

# ========================================================
# String : 5 questions
# ========================================================

Longest Substring Without Repeating Characters
Longest Repeating Character Replacement
Minimum Window Substring
Valid Anagram
Group Anagrams
Valid Parentheses
Valid Palindrome
Longest Palindromic Substring
Palindromic Substrings
Encode and Decode Strings (Leetcode Premium)


# ========================================================
# Tree : 5 questions
# ========================================================

Maximum Depth of Binary Tree
Same Tree
Invert/Flip Binary Tree
Binary Tree Maximum Path Sum
Binary Tree Level Order Traversal
Serialize and Deserialize Binary Tree
Subtree of Another Tree
Construct Binary Tree from Preorder and Inorder Traversal
Validate Binary Search Tree
Kth Smallest Element in a BST
Lowest Common Ancestor of BST
Implement Trie (Prefix Tree)
Add and Search Word
Word Search II

# ========================================================
# Heap : 5 questions
# ========================================================

Merge K Sorted Lists
Top K Frequent Elements
Find Median from Data Stream

