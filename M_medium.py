
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












