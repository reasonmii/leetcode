
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

        dummy = cur = ListNode(-1)

        while list1 and list2:
            if list1.val <= list2.val:
                cur.next = list1
                list1 = list1.next
            else:
                cur.next = list2
                list2 = list2.next
            cur = cur.next
        
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

# ======================================================================
# 88. Merge Sorted Array
# Topic : Dynamic Programming
# ======================================================================

class Solution(object):
    def merge(self, nums1, m, nums2, n):
        i = m-1
        j = n-1
        k = m+n-1

        while j >= 0:
            if i >= 0 and nums1[i] >= nums2[j]:
                nums1[k] = nums1[i]
                i -= 1
            else:
                nums1[k] = nums2[j]
                j -= 1
            k -=1

        return nums1

# ======================================================================
# 101. Symmetric Tree
# Topic : Tree
# ======================================================================

class Solution(object):
    
    def isMirror(self, left, right):
        if not left and not right:
            return True
        if not left or not right:
            return False
        return left.val == right.val and \
            self.isMirror(left.left, right.right) and \
            self.isMirror(left.right, right.left)

    
    def isSymmetric(self, root):
        if not root:
            return True
        return self.isMirror(root.left, root.right)

# ======================================================================
# 108. Convert Sorted Array to Binary Search Tree
# Topic : Binary Tree, Array, med
# ======================================================================

class Solution(object):
    def sortedArrayToBST(self, nums):
        if not nums:
            return

        med = len(nums) // 2

        return TreeNode(
            nums[med],
            self.sortedArrayToBST(nums[:med]),
            self.sortedArrayToBST(nums[med+1:])
        )

# ======================================================================
# 112. Path Sum
# Topic : Binary Tree, left, right
# ======================================================================

class Solution(object):
    def hasPathSum(self, root, targetSum):
        if not root:
            return False
        
        if not root.left and not root.right:
            return targetSum == root.val

        left = self.hasPathSum(root.left, targetSum - root.val)
        right = self.hasPathSum(root.right, targetSum - root.val)

        return left or right

# ======================================================================
# 118. Pascal's Triangle
# Topic : Pascal
# ======================================================================

class Solution(object):
    def generate(self, numRows):
        if numRows == 0:
            return []

        rst = [[1]]

        for i in range(1, numRows):
            prev = rst[-1]
            cur = [1]

            for j in range(1, i):
                cur.append(prev[j-1] + prev[j])
            cur.append(1)
            rst.append(cur)

        return rst

# ======================================================================
# 121. Best Time to Buy and Sell Stock
# Topic : min, max
# ======================================================================

class Solution(object):
    def maxProfit(self, prices):

        min_price = prices[0]
        profit = 0

        for p in prices:
            profit = max(p - min_price, profit)
            min_price = min(min_price, p)

        return profit

# ======================================================================
# 125. Valid Palindrome
# Topic : string, regular expression
# ======================================================================

class Solution(object):
    def isPalindrome(self, s):

        # sub(A, B, C) : if C is not in A, substitute it to B
        s = re.sub("[^a-z0-9]", "", s.lower()).replace(" ", "")
        return s == s[::-1]

# ======================================================================
# 160. Intersection of Two Linked Lists
# Topic : Linked List
# ======================================================================

class Solution(object):
    def getIntersectionNode(self, headA, headB):
        
        # speed : O(1)
        # 1) concatenate A and B : A+B and B+A
        # 2) Check if at some point, the 2 merged lists are pointing to the same node

        a, b = headA, headB
        while (a != b):
            a = headB if not a else a.next # 1 -> 8 -> 4 -> 5 -> NA -> 5  -> 6 -> 1
            b = headA if not b else b.next # 6 -> 1 -> 8 -> 4  -> 5 -> NA -> 4 -> 1
        return a # 8

# ======================================================================
# 168. Excel Sheet Column Title
# Topic : dic, chr, ord
# ======================================================================

class Solution(object):
    def convertToTitle(self, columnNumber):

        letters = [chr(x) for x in range(ord('A'), ord('Z')+1)] # unicode : A = 65
        rst = []

        while columnNumber > 0: # 701
            rst.append(letters[(columnNumber-1) % 26]) # 25 : Y
            columnNumber = (columnNumber-1) // 26 # 26 : Z 

        rst.reverse() # ZY
        return ''.join(rst)

# ======================================================================
# 169. Majority Element
# Topic : cnt
# ======================================================================

class Solution(object):
    def majorityElement(self, nums):

        cnt, maj = 0, 0

        for i in range(len(nums)):
            if cnt == 0 and maj != nums[i]:
                maj = nums[i]
                cnt += 1
            elif maj == nums[i]:
                cnt += 1
            else:
                # if current element is different
                # from the majority candidate
                cnt -= 1
        return maj

# ======================================================================
# 206. Reverse Linked List
# Topic : Linked List
# ======================================================================

class Solution(object):
    def reverseList(self, head):
        
        prev = None
        cur = head

        while cur:
            next_p = cur.next
            cur.next = prev
            prev = cur
            cur = next_p

        return prev

# ======================================================================
# 217. Contains Duplicate
# Topic : set
# ======================================================================

class Solution(object):
    def containsDuplicate(self, nums):
        return len(set(nums)) != len(nums)

# ======================================================================
# 225. Implement Stack using Queues
# Topic : Queue (collections - deque)
# ======================================================================

from collections import deque

class MyStack(object):

    def __init__(self):
        self.queue = []

    def push(self, x):
        self.queue.append(x)

    def pop(self):
        return self.queue.pop(0)
        
    def top(self):
        return self.queue[0]    

    def empty(self):
        return len(self.queue) == 0

# ======================================================================
# 232. Implement Queue using Stacks
# Topic : queue
# ======================================================================    

class MyQueue(object):

    def __init__(self):
        self.q = []

    def push(self, x):
        self.q.append(x)        

    def pop(self):
        return self.q.pop(0)

    def peek(self):
        return self.q[0]

    def empty(self):
        return len(self.q) == 0
        
# ======================================================================
# 242. Valid Anagram
# Topic : string, zip, setdefault
# ======================================================================    

class Solution(object):
    def isAnagram(self, s, t):

        if len(s) != len(t):
            return False

        dic = {}
    
        for sc, tc in zip(s, t):
            dic[sc] = dic.setdefault(sc, 0) + 1
            dic[tc] = dic.setdefault(tc, 0) - 1

        # check if there's at least one true value
        return not any(dic.values())

# ======================================================================
# 349. Intersection of Two Arrays
# Topic : Array
# ======================================================================    

class Solution(object):
    def intersection(self, nums1, nums2):

        return list(set([i for i in nums1 if i in nums2]))
        # return set(nums1).intersection(set(nums2))

# ======================================================================
# 367. Valid Perfect Square
# Topic : Binary search
# ======================================================================    

class Solution(object):
    def isPerfectSquare(self, num):

        if 0 <= num < 2:
            return True

        left, right = 1, num
        while left <= right:
            mid = (left + right) // 2
            if mid * mid == num:
                return True
            elif mid * mid < num:
                left = mid + 1
            else:
                right = mid - 1

        return  False            

# ======================================================================
# 496. Next Greater Element I
# Topic : stack
# ======================================================================

class Solution(object):
    def nextGreaterElement(self, nums1, nums2):

        dic = {}
        stk = []

        for i in range(len(nums2)):

            while i > 0 and stk and stk[-1] < nums2[i]:
                dic[stk[-1]] = nums2[i]
                stk.pop()
            
            stk.append(nums2[i])

        rst = []
        for n in nums1:
            if n in dic:
                rst.append(dic[n])
            else:
                rst.append(-1)

        return rst

# ======================================================================
# 509. Fibonacci Number
# Topic : fibonacci
# ======================================================================

class Solution(object):
    def fib(self, n):

        if n <= 1:
            return n

        dp = [0 for _ in range(n+1)]
        dp[1] = 1

        for i in range(2, n+1):
            dp[i] = dp[i-2] + dp[i-1]

        return dp[-1]

# ======================================================================
# 605. Can Place Flowers
# Topic : list
# ======================================================================

class Solution(object):
    def canPlaceFlowers(self, flowerbed, n):

        if n == 0:
            return True
        
        for i, p in enumerate(flowerbed):
            if p == 0 and \
            (i == 0 or flowerbed[i-1] == 0) and \
            (i == len(flowerbed)-1 or flowerbed[i+1]== 0):
                n -= 1
                flowerbed[i] = 1
                if n == 0:
                    return True
        return False

# ======================================================================
# 704. Binary Search
# Topic : binary search
# ======================================================================

class Solution(object):
    def search(self, nums, target):

        left, right = 0, len(nums)-1

        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid+1
            else:
                right = mid-1
        
        return -1

# ======================================================================
# 706. Design HashMap
# Topic : hash map
# ======================================================================

class MyHashMap(object):

    def __init__(self):
        self.data = [None] * 1000001   

    def put(self, key, value):
        self.data[key] = value        

    def get(self, key):
        val = self.data[key]
        return val if val != None else -1

    def remove(self, key):
        self.data[key] = None

# ======================================================================
# 733. Flood Fill
# Topic : Array, dfs, same as island (medium) question logic (200, 695)
# ======================================================================

class Solution(object):
    def floodFill(self, image, sr, sc, color):

        rows, cols = len(image), len(image[0])

        def dfs(row, col, original):
            if row < 0 or col < 0 or row >= rows or col >= cols or \
            image[row][col] != original or image[row][col] == color:
                return

            image[row][col] = color
            dfs(row-1, col, original)
            dfs(row+1, col, original)
            dfs(row, col-1, original)
            dfs(row, col+1, original)

        dfs(sr, sc, image[sr][sc])
        return image

# ======================================================================
# 859. Buddy Strings
# Topic : string
# ======================================================================

class Solution(object):
    def buddyStrings(self, s, goal):

        if len(s) != len(goal):
            return False
        
        if s == goal and len(s) != len(set(s)):
            return True

        diff_s, diff_g = [], []
        for i in range(len(s)):
            if s[i] != goal[i]:
                diff_s.append(s[i])
                diff_g.append(goal[i])

                if len(diff_s) > 2:
                    return False
        
        return len(diff_s) == 2 and sorted(diff_s) == sorted(diff_g)

# ======================================================================
# 1047. Remove All Adjacent Duplicates In String
# Topic : stack
# ======================================================================
        
class Solution(object):
    def removeDuplicates(self, s):

        rst = []
        for ch in s:
            if rst and ch == rst[-1]:
                rst.pop()
            else:
                rst.append(ch)

        return ''.join(rst)

# ======================================================================
# 1189. Maximum Number of Balloons
# Topic : text.count
# ======================================================================

class Solution(object):
    def maxNumberOfBalloons(self, text):

        return min(text.count('b'), text.count('a'), text.count('l') // 2, text.count('o') // 2, text.count('n'))

# ======================================================================
# 1661. Average Time of Process per Machine
# Topic : SQL
# ======================================================================

select a1.machine_id
     , round(sum(a2.timestamp - a1.timestamp) / count(*), 3) processing_time
from activity a1
left join activity a2
       on a1.machine_id = a2.machine_id
       and a1.process_id = a2.process_id
       and a2.activity_type = 'end'
where a1.activity_type = 'start'
group by a1.machine_id

# ======================================================================
# 1757. Recyclable and Low Fat Products
# Topic : SQL
# ======================================================================

select product_id
from products
where low_fats = 'Y' and recyclable = 'Y'

# ======================================================================
# 1768. Merge Strings Alternately
# Topic : index, pop
# ======================================================================

class Solution(object):
    def mergeAlternately(self, word1, word2):

        rst = ''
        idx = 0
        word1, word2 = list(word1), list(word2)

        while word1 and word2:
            if idx % 2 == 0 and word1:
                rst += word1.pop(0)
            else:
                rst += word2.pop(0)
            idx += 1

        return rst + ''.join(word1) if word1 else rst + ''.join(word2)

# ======================================================================
# 2235. Add Two Integers
# Topic : add
# ======================================================================

class Solution(object):
    def sum(self, num1, num2):
        return num1 + num2

# ======================================================================
# 2236. Root Equals Sum of Children
# Topic : binary tree, add
# ======================================================================

lass Solution(object):
    def checkTree(self, root):
        return root.val == (root.left.val + root.right.val)

# ======================================================================
# 2706. Buy Two Chocolates
# Topic : binary tree, add
# ======================================================================

class Solution(object):
    def buyChoco(self, prices, money):
        
        prices.sort()
        left = money - sum(prices[:2])
        return left if left >= 0 else money




