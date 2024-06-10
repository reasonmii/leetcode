# https://leetcode.com/discuss/general-discussion/460599/blind-75-leetcode-questions

# ========================================================
# Array : 10 questions
# ========================================================

# Two Sum (Easy)

class Solution(object):
    def twoSum(self, nums, target):

        rst = {}
        for i, n in enumerate(nums):
            diff = target - n
            if diff in rst:
                return [rst[diff], i]
            rst[n] = i

        return []

# Best Time to Buy and Sell Stock (Easy)

class Solution(object):
    def maxProfit(self, prices):

        min_pr = prices[0]
        max_pf = 0

        for p in prices[1:]:
            max_pf = max(max_pf, p - min_pr)
            min_pr = min(min_pr, p)

        return max_pf

# Contains Duplicate (Easy)

class Solution(object):
    def containsDuplicate(self, nums):
        return len(set(nums)) != len(nums)

# Product of Array Except Self (Medium)

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

# 53. Maximum Subarray

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

# Maximum Product Subarray


# Find Minimum in Rotated Sorted Array
# Search in Rotated Sorted Array
# 3 Sum
# Container With Most Water

# ========================================================
# Binary : 5 questions
# ========================================================

Sum of Two Integers
Number of 1 Bits
Counting Bits
Missing Number
Reverse Bits
Dynamic Programming
Climbing Stairs
Coin Change
Longest Increasing Subsequence
Longest Common Subsequence
Word Break Problem
Combination Sum
House Robber
House Robber II
Decode Ways
Unique Paths
Jump Game
Graph
Clone Graph
Course Schedule
Pacific Atlantic Water Flow
Number of Islands
Longest Consecutive Sequence
Alien Dictionary (Leetcode Premium)
Graph Valid Tree (Leetcode Premium)
Number of Connected Components in an Undirected Graph (Leetcode Premium)
Interval
Insert Interval
Merge Intervals
Non-overlapping Intervals
Meeting Rooms (Leetcode Premium)
Meeting Rooms II (Leetcode Premium)
Linked List
Reverse a Linked List
Detect Cycle in a Linked List
Merge Two Sorted Lists
Merge K Sorted Lists
Remove Nth Node From End Of List
Reorder List
Matrix
Set Matrix Zeroes
Spiral Matrix
Rotate Image
Word Search
String
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
Tree
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
Heap
Merge K Sorted Lists
Top K Frequent Elements
Find Median from Data Stream

