#include <iostream>
#include <omp.h>
#include <bits/stdc++.h>

using namespace std;

class Graph
{
public:
    int vertices;
    int edges;
    vector<vector<int>> graph;
    vector<bool> visited;

    // Function to generate a random graph
    vector<vector<int>> generateGraph(int n)
    {
        vector<vector<int>> graph(n);
        // Generate random connections between vertices
        for (int i = 0; i < n - 1; i++)
        {
            int neighbor = rand() % n;
            if (neighbor != i)
            {
                graph[i].push_back(neighbor);
                graph[neighbor].push_back(i);
            }
        }
        return graph;
    }

    // Constructor
    Graph(int n)
    {
        vertices = n;
        graph = generateGraph(vertices);
        edges = 0;
        for (const auto &adjList : graph)
        {
            edges += adjList.size();
        }
        edges /= 2; // Since it's undirected
        visited.assign(vertices, false);
    }

    void printGraph()
    {
        for (int i = 0; i < vertices; i++)
        {
            cout << i << " -> ";
            for (auto j = graph[i].begin(); j != graph[i].end(); j++)
            {
                cout << *j << " ";
            }
            cout << endl;
        }
    }

    void initialize_visited()
    {
        visited.assign(vertices, false);
    }

    void dfs(int i)
    {
        stack<int> s;
        s.push(i);
        visited[i] = true;

        while (s.empty() != true)
        {
            int current = s.top();
            cout << current << " ";
            s.pop();
            for (int neighbor : graph[current])
            {
                if (!visited[neighbor])
                {
                    s.push(neighbor);
                    visited[neighbor] = true;
                }
            }
        }
    }

    void parallel_dfs(int i)
    {
        stack<int> s;
        s.push(i);
        visited[i] = true;

        while (s.empty() != true)
        {
            int current = s.top();
            cout << current << " ";

#pragma omp critical // ensures only one thread accesses/modifies shared data like stack or visited at a time.
            s.pop();

#pragma omp parallel for // It parallelizes a for loop, meaning it divides the loop's iterations among multiple threads
            for (int i = 0; i < graph[current].size(); i++)
            {
                int neighbor = graph[current][i];
                if (!visited[neighbor])
                {
#pragma omp critical
                    {
                        s.push(neighbor);
                        visited[neighbor] = true;
                    }
                }
            }
        }
    }

    void bfs(int i)
    {
        queue<int> q;
        q.push(i);
        visited[i] = true;

        while (q.empty() != true)
        {
            int current = q.front();
            q.pop();
            cout << current << " ";
            for (int neighbor : graph[current])
            {
                if (!visited[neighbor])
                {
                    q.push(neighbor);
                    visited[neighbor] = true;
                }
            }
        }
    }

    void parallel_bfs(int i)
    {
        queue<int> q;
        q.push(i);
        visited[i] = true;

        while (q.empty() != true)
        {

            int current = q.front();
            cout << current << " ";

#pragma omp critical
            q.pop();

#pragma omp parallel for
            for (int i = 0; i < graph[current].size(); i++)
            {
                int neighbor = graph[current][i];
                if (!visited[neighbor])
                {

#pragma omp critical
                    {
                        q.push(neighbor);
                        visited[neighbor] = true;
                    }
                }
            }
        }
    }
};

int main(int argc, char const *argv[])
{
    int n = 100; // You can change the number of vertices here
    Graph g(n);

    cout << "Adjacency List:\n";
    g.printGraph();
    g.initialize_visited();

    cout << "Depth First Search: \n";
    auto start = chrono::high_resolution_clock::now();
    g.dfs(0);
    cout << endl;
    auto end = chrono::high_resolution_clock::now();
    cout << "Time taken: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds" << endl;

    cout << "Parallel Depth First Search: \n";
    g.initialize_visited();
    start = chrono::high_resolution_clock::now();
    g.parallel_dfs(0);
    cout << endl;
    end = chrono::high_resolution_clock::now();
    cout << "Time taken: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds" << endl;

    cout << "Breadth First Search: \n";
    g.initialize_visited();
    start = chrono::high_resolution_clock::now();
    g.bfs(0);
    cout << endl;
    end = chrono::high_resolution_clock::now();
    cout << "Time taken: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds" << endl;

    cout << "Parallel Breadth First Search: \n";
    g.initialize_visited();
    start = chrono::high_resolution_clock::now();
    g.parallel_bfs(0);
    cout << endl;
    end = chrono::high_resolution_clock::now();
    cout << "Time taken: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds" << endl;

    return 0;
}
