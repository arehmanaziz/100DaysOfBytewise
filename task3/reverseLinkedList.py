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


    def reverseLL(self):
        temp = self.head
        prev = None
        while temp:
            Next = temp.next
            temp.next = prev
            prev = temp
            temp = Next
            
        self.head = prev


if __name__ == '__main__':

    values1 = [1, 2, 3, 4, 5]
    llist = LinkedList()
    llist.addList(values1)
    print(llist)

    llist.reverseLL()
    print(llist)
    