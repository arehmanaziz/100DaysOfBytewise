class Node:
    def __init__(self, value):
        self.value = value
        self.next = None


class LinkedList:
    def __init__(self):
        self.head = None


    def __str__(self):
        temp = self.head
        values = []
        while temp:
            values.append(str(temp.value))
            temp = temp.next
        return " -> ".join(values)


    def addList(self, values_list):
        self.head = Node(values_list[0])
        temp = self.head
        for i in range(1, len(values_list)):
            temp.next = Node(values_list[i])
            temp = temp.next

    
    def checkPalindrome(self):
        temp = self.head
        values = []
        while temp:
            values.append(temp.value)
            temp = temp.next
        if values == values[::-1]:
            print("The linked list is a palindrome")
        else:
            print("The linked list is not a palindrome")
            


if __name__ == '__main__':

    values1 = [1, 2, 3, 2, 1]
    llist1 = LinkedList()
    llist1.addList(values1)
    print(llist1)
    llist1.checkPalindrome()
    
    print()

    values2 = [1, 2, 3, 4, 5]
    llist2 = LinkedList()
    llist2.addList(values2)
    print(llist2)
    llist2.checkPalindrome()