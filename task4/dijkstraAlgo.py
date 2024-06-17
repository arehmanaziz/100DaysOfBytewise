def dijkstra(graph, start):
    
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    priority_queue = [(0, start)]
    visited = set()
    
    while priority_queue:
        
        priority_queue.sort()
        current_distance, current_node = priority_queue.pop(0)

        if current_node in visited:
            continue
        
        visited.add(current_node)
        
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                priority_queue.append((distance, neighbor))
    
    return distances

if __name__ == '__main__':

    graph = {
        'A': {'B': 1, 'C': 4},
        'B': {'C': 2, 'D': 5},
        'C': {'D': 1},
        'D': {}
    }

    start_node = 'A'
    shortest_paths = dijkstra(graph, start_node)
    print(shortest_paths)
