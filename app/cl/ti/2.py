# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def getNumber(self, l1: ListNode | None): 
        if l1 is None:
            return 0
        temp=''
        while True: 
            temp+=str( l1.val)
            if l1.next is None:
                break
            else:
                l1=l1.next 
        temp=list(temp)
        temp.reverse()
        return int(''.join(temp))
    def number2ListNode(self,n:int):
        temp=list(str(n))
        temp.reverse()
        l1=ListNode(int(temp[0]))
        l2=l1
        for i in range(1,len(temp)):
            l2.next=ListNode(int(temp[i]),None)
            l2=l2.next
        return l1
    def printListNode(self,l1: ListNode | None):
        while l1 is not None:
            print(l1.val,end=' ')
            l1=l1.next
        print() 
    def addTwoNumbers2(self, l1: ListNode | None, l2: ListNode | None) -> ListNode | None:  
        carry = 0
        head = ListNode(0)
        new_node=head
        while  l1 is not None or l2 is not None:
            if l1 is not None and l2 is not None:
                new_node.next=ListNode((l1.val+l2.val+carry) % 10)
                carry = (l1.val+l2.val+carry) // 10
                l1 = l1.next
                l2 = l2.next
            elif l1 is not None:
                new_node.next=ListNode((l1.val+carry) % 10)
                carry = (l1.val+carry) // 10
                l1 = l1.next
                
            elif l2 is not None:
                new_node.next=ListNode((l2.val+carry) % 10)
                carry = (l2.val+carry) // 10
                l2 = l2.next
                
            new_node = new_node.next
        if carry == 1:
            new_node.next=ListNode(1)
        return (head.next)

    def addTwoNumbers(self, l1: ListNode | None, l2: ListNode | None) -> ListNode | None: 
        dummy=ListNode()
        cur=dummy
        carry=0
        while l1 or l2 or carry:
            s=carry
            if l1:
                s+=l1.val
                l1=l1.next
            if l2:
                s+=l2.val
                l2=l2.next
            cur.next=ListNode(s%10)
            cur=cur.next
            carry=s//10
        return dummy.next



       

         
if __name__ == '__main__':
    s=Solution()
    l1=s.number2ListNode(99999)
    l2=s.number2ListNode(12)
    l3=s.addTwoNumbers2(l1,l2)
    s.printListNode(l3) 

        