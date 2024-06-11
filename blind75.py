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

# 322. Coin Change (Medium) ##

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

# Longest Increasing Subsequence
Longest Common Subsequence
Word Break Problem
Combination Sum
House Robber
House Robber II
Decode Ways
Unique Paths
Jump Game

# ========================================================
# Graph : 5 questions
# ========================================================

Clone Graph
Course Schedule
Pacific Atlantic Water Flow
Number of Islands
Longest Consecutive Sequence
Alien Dictionary (Leetcode Premium)
Graph Valid Tree (Leetcode Premium)
Number of Connected Components in an Undirected Graph (Leetcode Premium)


# ========================================================
# Interval : 5 questions
# ========================================================

Insert Interval
Merge Intervals
Non-overlapping Intervals
Meeting Rooms (Leetcode Premium)
Meeting Rooms II (Leetcode Premium)


# ========================================================
# Linked List : 5 questions
# ========================================================


Reverse a Linked List
Detect Cycle in a Linked List
Merge Two Sorted Lists
Merge K Sorted Lists
Remove Nth Node From End Of List
Reorder List

# ========================================================
# Matrix : 5 questions
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

