class Node:

    def __init__(self, value):

        self.value = value

        self.left = None

        self.right = None


class BST:

    def __init__(self):

        self.root = None



    def inorder(self, root):
        if root:
            self.inorder(root.left)

            print(root.value, end=" ")

            self.inorder(root.right)

    

    def insert(self, value):

        newNode = Node(value)

        if self.root is None:

            self.root = newNode

        else:

            temp = self.root

            while True:

                if value < temp.value:

                    if temp.left is None:

                        temp.left = newNode

                        break
                    else:

                        temp = temp.left
                else:

                    if temp.right is None:

                        temp.right = newNode

                        break
                    else:

                        temp = temp.right

    

    def remove(self, root, value):

        if root is None:
            return root

        if value < root.value:
            root.left = self.remove(root.left, value)

        elif value > root.value:

            root.right = self.remove(root.right, value)

        else:

            if root.left is None:

                return root.right

            elif root.right is None:

                return root.left


            root.value = self.minValue(root.right)


            root.right = self.remove(root.right, root.value)


        return root



    def minValue(self, root):


        minv = root.value


        while root.left:

            minv = root.left.value

            root = root.left


        return minv




if __name__ == '__main__':

    bst = BST()

    bst.insert(50)

    bst.insert(30)

    bst.insert(20)

    bst.insert(40)

    bst.insert(70)

    bst.insert(60)

    bst.insert(80)

    bst.inorder(bst.root)

    print()

    bst.remove(bst.root, 20)
    bst.inorder(bst.root)

    print()

    bst.remove(bst.root, 30)
    bst.inorder(bst.root)

    print()

    bst.remove(bst.root, 50)
    bst.inorder(bst.root)

    print()

    bst.remove(bst.root, 60)
    bst.inorder(bst.root)

    print()

    bst.remove(bst.root, 70)
    bst.inorder(bst.root)

    print()

    bst.remove(bst.root, 80)
    bst.inorder(bst.root)

    print()

    bst.remove(bst.root, 40)

