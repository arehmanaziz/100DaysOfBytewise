from collections import defaultdict

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)


    def addEdge(self, u, v):
        self.graph[u].append(v)


    def BFS(self, s):

        visited = [False] * (max(self.graph) + 1)

        queue = []
        ans = []
        queue.append(s)
        visited[s] = True

        while queue:
            s = queue.pop(0)
            ans.append(s)
            for i in self.graph[s]:
                if visited[i] == False:
                    queue.append(i)
                    visited[i] = True

        return ans


    def DFS(self, s):

        ans = []

        def DFSFunc(self, v, visited):

            ans.append(v)
            visited[v] = True
            for i in self.graph[v]:
                if visited[i] == False:
                    DFSFunc(self, i, visited)

        
        visited = [False] * (max(self.graph) + 1)
        DFSFunc(self, s, visited)

        return ans


if __name__ == "__main__":

    g = Graph()
    g.addEdge(0, 1)
    g.addEdge(0, 2)
    g.addEdge(1, 2)
    g.addEdge(2, 0)
    g.addEdge(2, 3)
    g.addEdge(3, 3)

    
    bfs = g.BFS(2)
    print("Breath First Search with starting node 2")
    print(bfs)
    print()

    dfs = g.DFS(2)
    print("Depth First Search with starting node 2")
    print(dfs)
