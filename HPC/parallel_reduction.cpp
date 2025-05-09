#include <iostream>
#include <vector>
#include <omp.h>
#include <climits> // For INT_MAX, INT_MIN
using namespace std;

// Parallel reduction to calculate the minimum value
int parallel_min(const vector<int> &arr)
{
    int min_value = INT_MAX;

#pragma omp parallel for reduction(min : min_value)
    for (int i = 0; i < arr.size(); ++i)
    {
        min_value = min(min_value, arr[i]);
    }

    return min_value;
}

// Parallel reduction to calculate the maximum value
int parallel_max(const vector<int> &arr)
{
    int max_value = INT_MIN;

#pragma omp parallel for reduction(max : max_value)
    for (int i = 0; i < arr.size(); ++i)
    {
        max_value = max(max_value, arr[i]);
    }

    return max_value;
}

// Parallel reduction to calculate the sum
int parallel_sum(const vector<int> &arr)
{
    int sum_value = 0;

#pragma omp parallel for reduction(+ : sum_value)
    for (int i = 0; i < arr.size(); ++i)
    {
        sum_value += arr[i];
    }

    return sum_value;
}

// Parallel reduction to calculate the average
double parallel_average(const vector<int> &arr)
{
    int sum_value = parallel_sum(arr);
    return sum_value / arr.size();
}

int main()
{
    // Example vector of numbers
    vector<int> arr = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // Set the number of threads for OpenMP
    omp_set_num_threads(4); // You can set this dynamically or based on the system's cores.

    // Perform the operations using parallel reduction
    int min_val = parallel_min(arr);
    int max_val = parallel_max(arr);
    int sum_val = parallel_sum(arr);
    double avg_val = parallel_average(arr);

    // Output the results
    cout << "Min: " << min_val << endl;
    cout << "Max: " << max_val << endl;
    cout << "Sum: " << sum_val << endl;
    cout << "Average: " << avg_val << endl;

    return 0;
}

// g++ -fopenmp parallel_reduction.cpp -o parallel_reduction
// ./parallel_reduction