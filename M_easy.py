
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
























