
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
# 14. Longest Common Prefix
# Topic : string, Trie
# ======================================================================

class Solution(object):
    def longestCommonPrefix(self, strs):

        if not strs: return None
        
        rst = strs[0]
        for word in strs[1:]:
            i = 0
            while i < len(rst) and i < len(word):
                if rst[i] != word[i]:
                    break
                i += 1
            rst = rst[:i]

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
# 26. Remove Duplicates from Sorted Array
# Topic : Array, two pointers
# ======================================================================

class Solution(object):
    def removeDuplicates(self, nums):

        k = 0
        for n in nums:
            if nums[k] != n:
                k += 1
                nums[k] = n
        return k+1

# ======================================================================
# 27. Remove Element
# Topic : Array, two pointers
# ======================================================================

class Solution(object):
    def removeElement(self, nums, val):

        k, i = 0, 0
        while i < len(nums):
            if nums[i] != val:
                nums[k] = nums[i]
                k += 1
            i += 1
        return k

# ======================================================================
# 28. Find the Index of the First Occurrence in a String
# Topic : string, index
# ======================================================================

class Solution(object):
    def strStr(self, haystack, needle):

        n = len(needle)
        for i, ch in enumerate(haystack):
            if haystack[i:i+n] == needle:
                return i
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
# 69. Sqrt(x)
# Topic : Binary Search
# ======================================================================

class Solution(object):
    def mySqrt(self, x):

        l, r = 0, x

        while l <= r:
            m = (l+r) // 2
            if m * m <= x < (m+1) * (m+1):
                return m
            elif m * m < x:
                l = m+1
            else:
                r = m-1

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
# 94. Binary Tree Inorder Traversal
# Topic : Tree
# ======================================================================

class Solution(object):
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """

        rst = []
        self.inOrder(root, rst)
        return rst

    def inOrder(self, root, rst):
        if not root:
            return

        self.inOrder(root.left, rst)
        rst.append(root.val)
        self.inOrder(root.right, rst)

        return rst

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
# 110. Balanced Binary Tree
# Topic : Binary Tree
# ======================================================================

class Solution(object):
    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """

        return self.balance(root) != -1

    def balance(self, root):
        if not root:
            return 0

        left = self.balance(root.left)
        right = self.balance(root.right)

        if left == -1 or right == -1 or abs(left - right) > 1:
            return -1
        
        return 1 + max(left, right)

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
# 136. Single Number
# Topic : Array, bit
# ======================================================================

class Solution(object):
    def singleNumber(self, nums):

        # linear runtime
        # use only constant extra space

        # ^= : XOR (exclusive OR)
        # commonly used for finding the single non-repeating element in an array
        # when all other elements occur in pairs

        for i in range(1, len(nums)): # [4, 1, 2, 1, 2]
            # any number XOR with itself is 0
            # so each duplicate number will cancel out itself
            nums[0] ^= nums[i] # 4 -> 5 -> 7 -> 6 -> 4

        return nums[0]

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
# 171. Excel Sheet Column Number
# Topic : dic
# ======================================================================

class Solution(object):
    def titleToNumber(self, columnTitle):

        res = 0
        val = [i for i in range(1, 27)]
        letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        d = dict(zip(letters, val))

        for col in columnTitle:
            res = res * 26 + d[col]
        return res

# ======================================================================
# 202. Happy Number
# Topic : set
# ======================================================================

class Solution(object):
    def isHappy(self, n):
        
        nums = set()

        while n != 1:
            if n in nums:
                return False
            nums.add(n)
            n = sum([int(i) ** 2 for i in str(n)])
        else:
            return True

# ======================================================================
# 205. Isomorphic Strings
# Topic : string
# ======================================================================

class Solution(object):
    def isIsomorphic(self, s, t):
        
        return len(set(zip(s,t))) == len(set(s)) == len(set(t))

# ======================================================================
# 206. Reverse Linked List
# Topic : Linked List
# ======================================================================

class Solution(object):
    def reverseList(self, head):
        
        prev = None

        while head:
            next_p = cur.next
            head.next = prev
            prev = head
            head = next_p

        return prev

# ======================================================================
# 217. Contains Duplicate
# Topic : set
# ======================================================================

class Solution(object):
    def containsDuplicate(self, nums):
        return len(set(nums)) != len(nums)


# ======================================================================
# 219. Contains Duplicate II
# Topic : dic
# ======================================================================

class Solution(object):
    def containsNearbyDuplicate(self, nums, k):

        dic = {}
        for i, n in enumerate(nums):
            if n in dic and i - dic[n] <= k:
                return True
            else:
                dic[n] = i
        return False

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
# 231. Power of Two
# Topic : math
# ======================================================================    

class Solution(object):
    def isPowerOfTwo(self, n):
        
        if n <= 0:
            return False

        while n % 2 == 0:
            n /= 2

        return n == 1
        
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
# 234. Palindrome Linked List
# Topic : Linked List
# ======================================================================    

class Solution(object):
    def isPalindrome(self, head):

        ### Find mid-point
        slow, fast = head, head

        while fast and fast.next:
            slow, fast = slow.next, fast.next.next

        ### Reverse
        prev = None

        while slow:
            next_p = slow.next
            slow.next = prev
            prev = slow
            slow = next_p

        while prev:
            if prev.val != head.val:
                return False
            prev, head = prev.next, head.next
        
        return True

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
# 257. Binary Tree Paths
# Topic : Binary Tree
# ======================================================================    

class Solution(object):
    def binaryTreePaths(self, root):

        def dfs(node, path, rst):
            if not node:
                return

            path += str(node.val)
            if not node.left and not node.right:
                rst.append(path)
            else:
                dfs(node.left, path+'->', rst)
                dfs(node.right, path+'->', rst)

        rst = []
        dfs(root, '', rst)
        return rst

# ======================================================================
# 268. Missing Number
# Topic : range
# ======================================================================    

class Solution(object):
    def missingNumber(self, nums):
        
        return sum(range(len(nums)+1)) - sum(nums)
        
# ======================================================================
# 283. Move Zeroes
# Topic : list, index
# ======================================================================    

class Solution(object):
    def moveZeroes(self, nums):

        idx = 0
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[i], nums[idx] = nums[idx], nums[i]
                idx += 1

# ======================================================================
# 345. Reverse Vowels of a String
# Topic : String
# ======================================================================    

class Solution(object):
    def reverseVowels(self, s):

        i, j = 0, len(s)-1
        s = list(s)

        while i < j:
            if s[i].lower() in 'aeiou' and s[j].lower() in 'aeiou':
                s[i], s[j] = s[j], s[i]
                i += 1
                j -= 1
            elif s[i].lower() not in 'aeiou' and s[j].lower() in 'aeiou':
                i += 1
            elif s[j].lower() not in 'aeiou' and s[i].lower() in 'aeiou':
                j -= 1
            else:
                i += 1
                j -= 1

        return ''.join(s)

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
# 387. First Unique Character in a String
# Topic : String
# ======================================================================    

class Solution(object):
    def firstUniqChar(self, s):

        dic = Counter(s)
        for i, c in enumerate(s):
            if dic[c] == 1:
                return i
        return -1

# ======================================================================
# 392. Is Subsequence
# Topic : String
# ======================================================================    

class Solution(object):
    def isSubsequence(self, s, t):

        i = 0
        for ch in t:
            if i < len(s) and s[i] == ch:
                i += 1

        return i == len(s)

# ======================================================================
# 412. Fizz Buzz
# Topic : List, Math
# ======================================================================

class Solution(object):
    def fizzBuzz(self, n):

        rst = []
        for i in range(1, n+1):
            if i % 3 == 0 and i % 5 == 0:
                rst.append("FizzBuzz")
            elif i % 3 == 0:
                rst.append('Fizz')
            elif i % 5 == 0:
                rst.append('Buzz')
            else:
                rst.append(str(i))

        return rst

# ======================================================================
# 461. Hamming Distance
# Topic : bit
# ======================================================================

class Solution(object):
    def hammingDistance(self, x, y):
        return bin(x^y).count('1') # x^y : XOR

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
# 543. Diameter of Binary Tree
# Topic : binary tree
# ======================================================================

class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:

        d = 0

        def longestPath(node):
            if not node: return 0

            nonlocal d

            left = longestPath(node.left)
            right = longestPath(node.right)
            
            d = max(d, left+right)
            
            return max(left, right) + 1          

        longestPath(root)
        return d

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
# 628. Maximum Product of Three Numbers
# Topic : list
# ======================================================================

class Solution(object):
    def maximumProduct(self, nums):

        nums.sort()
        n1 = nums[-3] * nums[-2] * nums[-1]
        n2 = nums[0] * nums[1] * nums[-1]

        return max(n1, n2)

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
# 724. Find Pivot Index
# Topic : list
# ======================================================================

class Solution(object):
    def pivotIndex(self, nums):

        left = 0
        right = sum(nums)

        for i, n in enumerate(nums):
            right -= n
            if left == right:
                return i
            left += n

        return -1

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
# 746. Min Cost Climbing Stairs
# Topic : Daynamic Programming
# ======================================================================

class Solution(object):
    def minCostClimbingStairs(self, cost):

        step1 = cost[-2]
        step2 = cost[-1]

        for i in range(len(cost)-3, -1, -1):
            tmp = step1

            if step1 < step2:
                step1 += cost[i]
            else:
                step1 = step2 + cost[i]
            
            step2 = tmp

        return min(step1, step2)

# ======================================================================
# 836. Rectangle Overlap
# Topic : string
# ======================================================================

class Solution(object):
    def isRectangleOverlap(self, rec1, rec2):

        if rec1[2] <= rec2[0] or rec2[2] <= rec1[0]:
            return False
        
        if rec1[3] <= rec2[1] or rec2[3] <= rec1[1]:
            return False

        return True

# ======================================================================
# 859. Buddy Strings
# Topic : string
# ======================================================================

class Solution(object):
    def buddyStrings(self, s, goal):

        if len(s) != len(goal):
            return False

        if len(s) != len(set(s)) and s == goal:
            return True

        sl, gl = [], []

        for sc, gc in zip(s, goal):
            if sc != gc:
                sl.append(sc)
                gl.append(gc)
                if len(sl) > 2:
                    return False

        return len(sl) == 2 and sorted(sl) == sorted(gl)

# ======================================================================
# 905. Sort Array By Parity
# Topic : list
# ======================================================================

class Solution(object):
    def sortArrayByParity(self, nums):
        
        n1 = []
        n2 = []
        for n in nums:
            if n % 2 == 1:
                n2.append(n)
            else:
                n1.append(n)

        return n1 + n2

# ======================================================================
# 993. Cousins in Binary Tree
# Topic : Tree
# ======================================================================

class Solution(object):
    def isCousins(self, root, x, y):

        rst = []

        def dfs(node, parent, depth):
            if not node:
                return
            
            if node.val == x or node.val == y:
                rst.append((parent, depth))
            
            dfs(node.left, node, depth+1)
            dfs(node.right, node, depth+1)

        dfs(root, None, 0)

        node_x, node_y = rst
        return node_x[0] != node_y[0] and node_x[1] == node_y[1]

# ======================================================================
# 997. Find the Town Judge
# Topic : List
# ======================================================================

class Solution(object):
    def findJudge(self, n, trust):
        
        person = [0] * (n+1)
        judge = [0] * (n+1)

        for p, j in trust:
            person[p] += 1
            judge[j] += 1
        
        for i in range(1, n+1):
            if person[i] == 0 and judge[i] == n-1:
                return i
        return -1

# ======================================================================
# 1002. Find Common Characters
# Topic : Character
# ======================================================================

class Solution(object):
    def commonChars(self, words):

        def count(word):
            freq = [0] * 26
            for ch in word:
                freq[ord(ch) - ord('a')] += 1
            return freq
        
        def inter(freq1, freq2):
            return [min(f1, f2) for f1, f2 in zip(freq1, freq2)]

        cnt = count(words[0])

        for word in words[1:]:
            cnt = inter(cnt, count(word))

        rst = []
        for i in range(26):
            rst.extend([chr(i + ord('a'))] * cnt[i])

        return rst

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
# 1071. Greatest Common Divisor of Strings
# Topic : string
# ======================================================================

class Solution(object):
    def gcdOfStrings(self, str1, str2):

        if str1 + str2 != str2 + str1:
            return ""

        if len(str1) == len(str2):
            return str1
        
        if len(str1) > len(str2):
            return self.gcdOfStrings(str1[len(str2):], str2)
        
        return self.gcdOfStrings(str1, str2[len(str1):])
        
# ======================================================================
# 1189. Maximum Number of Balloons
# Topic : text.count
# ======================================================================

class Solution(object):
    def maxNumberOfBalloons(self, text):

        return min(text.count('b'), text.count('a'), text.count('l') // 2, text.count('o') // 2, text.count('n'))

# ======================================================================
# 1207. Unique Number of Occurrences
# Topic : count
# ======================================================================

class Solution(object):
    def uniqueOccurrences(self, arr):
        
        rst = Counter(arr).values()
        return len(rst) == len(set(rst))

# ======================================================================
# 1539. Kth Missing Positive Number
# Topic : count
# ======================================================================

class Solution(object):
    def findKthPositive(self, arr, k):
        j = 0
        for i in range(1, arr[-1]+1):
            if arr[j] > i:
                k -= 1
            else:
                j += 1
            if k == 0:
                return i
        return arr[-1] + k

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
        for w1 in word1:
            rst += w1
            for w2 in word2:
                rst += w2
                word2 = word2[1:]
                break

        return rst + word2

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




