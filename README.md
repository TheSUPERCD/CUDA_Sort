### Introduction

This project explores a massively parallel computation methodology using the famous **CUDA** framework from NVIDIA. In this project, we have evaluted three distinct algorthmic approches to sorting a randomly generated array of random integers, where the length of the random array was taken to be dynamic - as it is designed to be given to our program at runtime.

The aforementioned three algorthms are:

- Thrust library's default sort implementation (thrust.cu)

- Decimal digit-wise radix sort algorithm implemented to run on a single thread of an NVIDIA GPU (singlethread.cu)

- A multi-threaded, massively parallel, bit-wise radix sort implementation to run on an NVIDIA GPU (multithread.cu)

The source codes for all three can be built and run using the default `make` command in the terminal window:

```shell
$ make clean
$ make
nvcc -O3 -Xcompiler -Wall thrust.cu -o thrust
nvcc -O3 -Xcompiler -Wall singlethread.cu -o singlethread
nvcc -O3 -Xcompiler -Wall multithread.cu -o multithread
```

This will compile the source codes into three separate executables - `thrust`, `singlethread` and `multithread`.

When we run these programs against varying sizes of random arrays, obtain the general behavior and runtimes of the implementations, and plot them in graphs to get a better understanding of them ---- we can then compare them to see which implementation performs better, and then try to understand the reason behind this difference.

### Thrust Library Implementation

##### Description

Thrust is a C++ STL-based template library for the CUDA framework. Thrust allows us to implement high performance parallel computation systems while requiring minimal amounts of programming effort using a high-level interface to interact with the GPU device.

Thrust additionally provides us a rich collection of parallel algorithmic methods and functions. We too, have implemented our program using one of them in this assignment - `thrust::sort`.

```shell
$ ./thrust 100 10 1
Total time in seconds: 6.144e-06

The sorted array is:
[0, 0, 3, 7, 7, 8, 12, 12, 12, 13, 14, 17, 17, 17, 18, 20, 20, 20, 21, 21, 21, 23, 23, 23, 23, 23, 24, 25, 26, 27, 27, 27, 28, 28, 33, 35, 35, 37, 37, 38, 38, 43, 44, 44, 45, 45, 45, 47, 47, 47, 52, 54, 55, 56, 57, 58, 58, 60, 60, 60, 62, 65, 65, 65, 66, 67, 68, 68, 69, 70, 71, 71, 71, 72, 73, 73, 75, 76, 77, 77, 78, 78, 81, 81, 81, 82, 83, 84, 85, 87, 87, 88, 88, 88, 92, 92, 93, 95, 95, 97], size=100
```

##### Algorithm Implementation

The thrust library's sorting function is implemented using a parallel radix sort method called `cub radix sort`. This sorting algorithm extracts all the array elements' bits, and then groups them together while rearranging them for each bit position. This gives us a sorted array which can be made parallel as there is no sequential comparisons made to sort the array.

##### Time Complexity Analysis

The radix sort method's time complexity is dependent on the maximum possible values in the array. If that maximum possible value in the array is $k$, we can get the time complexity of the algorithm as $O((n+2)log_2(k))$ for bit-wise sorting. For a more general case, it would be: $O((n+b)log_b(k))$ for a sorting base = b. Our implementation assumes it is optimized for binary sorting, and hence the final time complexity becomes: 

***Time Complexity:*** $O((n+2)log_2(k))$

##### Performance

We run the ***thrust*** program with varying array sizes and plot their runtimes against the number of array elements. This gives us a clear understanding of the implementations performance metric, as the array size changes. We can also verify the results of our time complexity analysis from this graph.

| Number of Array Elements | Program Runtime (seconds) |
| ------------------------ | ------------------------- |
| 10000                    | 0.000116736               |
| 20000                    | 0.000123392               |
| 30000                    | 0.0001792                 |
| 40000                    | 0.00023152                |
| 50000                    | 0.000345536               |
| 60000                    | 0.000383136               |
| 70000                    | 0.000506592               |
| 80000                    | 0.000590784               |
| 90000                    | 0.00065808                |
| 100000                   | 0.000765952               |
| 200000                   | 0.00148474                |
| 300000                   | 0.00220058                |
| 400000                   | 0.0029399                 |
| 500000                   | 0.00358899                |
| 600000                   | 0.00486752                |
| 700000                   | 0.00595866                |
| 800000                   | 0.0076408                 |
| 900000                   | 0.00706662                |
| 1000000                  | 0.00762259                |
| 2000000                  | 0.0155263                 |
| 3000000                  | 0.023934                  |

The corresponding graph for the performance measurement is as follows:

<div align="center">
<img src="Thrust_Performance.png" />
</div>

##### Explanation

The thrust library's implementation of the sorting function is extremely optimized, reducing the number of kernel calls and saving the GPU memory to make the operation as efficient as possible for a general case scenario. This is why the plot appears almost like a straight line. But if we pay closer attention to the values obtained for array sizes in the millions (last three entries in the table), we can clearly see the $nlog(n)$ trend being displayed there. This is in line with our time complexity analysis.

### Single-Thread Implementation

##### Description

Radix sort is a linear sorting algorithm mostly used for sorting integer values. In this sorting algorithm, we perform a digit by digit sorting that starts from the least significant digit and goes up to the most significant digit of the array elements. The working principle of this algorithm is similar to the lexicographic arrange of words in a dictionary. They are not sorted by their entire length or overall letter arrangement. Instead, they are sorted using a letter by letter sorting method.

```shell
$ ./singlethread 100 10 1
Total time in seconds: 0.00062544

The sorted array is:
[0, 0, 3, 7, 7, 8, 12, 12, 12, 13, 14, 17, 17, 17, 18, 20, 20, 20, 21, 21, 21, 23, 23, 23, 23, 23, 24, 25, 26, 27, 27, 27, 28, 28, 33, 35, 35, 37, 37, 38, 38, 43, 44, 44, 45, 45, 45, 47, 47, 47, 52, 54, 55, 56, 57, 58, 58, 60, 60, 60, 62, 65, 65, 65, 66, 67, 68, 68, 69, 70, 71, 71, 71, 72, 73, 73, 75, 76, 77, 77, 78, 78, 81, 81, 81, 82, 83, 84, 85, 87, 87, 88, 88, 88, 92, 92, 93, 95, 95, 97], size=100
```

##### Algorithm Implementation

To implement this radix sort algorithm, we simply iterate through all the digit positions for every array element. The maximum number of digit positions are determined by the maximum possible value that the randomly generated array elements can possibly have. For each digit position, we create a bucket list to hold the count how many times each digit (0-9) occurs in all the array elements. Next, we rearrange the array elements in the order of how their digits in that specific digit position should be ordered by using a simple prefix scan on the bucket list followed by a counting sort.

##### Time Complexity Analysis

The radix sort method's time complexity is dependent on the maximum possible values in the array. If that maximum possible value in the array is $k$, we can get the time complexity of the algorithm as $O((n+10)log_10(k))$ for bit-wise sorting. Our implementation assumes it is optimized for binary sorting, and hence the final time complexity becomes: 

***Time Complexity:*** $O((n+10)log_10(k))$

##### Performance

We run the ***singlethread*** program with varying array sizes and plot their runtimes against the number of array elements. This gives us a clear understanding of the implementations performance metric, as the array size changes. We can also verify the results of our time complexity analysis from this graph.

| Number of Array Elements | Program Runtime (seconds) |
| ------------------------ | ------------------------- |
| 1000                     | 0.00357686                |
| 2000                     | 0.00883405                |
| 3000                     | 0.013045                  |
| 4000                     | 0.0172358                 |
| 5000                     | 0.0214361                 |
| 6000                     | 0.0256588                 |
| 7000                     | 0.029872                  |
| 8000                     | 0.0340629                 |
| 9000                     | 0.0382597                 |
| 10000                    | 0.042493                  |
| 20000                    | 0.105533                  |
| 30000                    | 0.141345                  |
| 40000                    | 0.166599                  |
| 50000                    | 0.200987                  |
| 60000                    | 0.241104                  |
| 70000                    | 0.282655                  |
| 80000                    | 0.32576                   |
| 90000                    | 0.3688                    |
| 100000                   | 0.411608                  |
| 200000                   | 1.05114                   |
| 300000                   | 1.51276                   |

The corresponding graph for the performance measurement is as follows:

<div align="center">
<img src="Single-Thread_Performance.png" />
</div>

##### Explanation

This single-thread implementation of the sorting function is extremely ill-optimized, as we are running it on a single thread, and that too on a GPU device with a much lower clock speeds compared to CPU devices, which causes our program to slow down tremendously for large array sizes. Here, we can clearly observe the time complexity trend ($O((n+10)log_10(k))$) being displayed in the plot.

### Multi-Thread Implementation

##### Algorithm Implementation

Our implementation of the multi-threaded program also uses the radix sort algorithm. Since the most efficient way of achieving a massively parallel sorting algorithm is to process all data points independently, we choose the most suitable method of doing it - using a bit-wise radix sort algorithm.

```shell
$ ./multithread 100 10 1
Total time in seconds: 0.000700096

The sorted array is:
[0, 0, 3, 7, 7, 8, 12, 12, 12, 13, 14, 17, 17, 17, 18, 20, 20, 20, 21, 21, 21, 23, 23, 23, 23, 23, 24, 25, 26, 27, 27, 27, 28, 28, 33, 35, 35, 37, 37, 38, 38, 43, 44, 44, 45, 45, 45, 47, 47, 47, 52, 54, 55, 56, 57, 58, 58, 60, 60, 60, 62, 65, 65, 65, 66, 67, 68, 68, 69, 70, 71, 71, 71, 72, 73, 73, 75, 76, 77, 77, 78, 78, 81, 81, 81, 82, 83, 84, 85, 87, 87, 88, 88, 88, 92, 92, 93, 95, 95, 97], size=100
```

##### Time Complexity

The radix sort algorithm operates on the basis of sorting without directly comparing the array elements (i.e. non-comparison type sorting). The thrust STL library's implementation of parallel sorting algorithms also uses this algorithm, albeit a more general case one which can sort any data-type in a massively parallel way. Since we used the same algorithm as the Thrust-libary's implementation, we can say that our method's time complexity is the same as the one mentioned in the thrust library's sort function's time complexity.

***Time Complexity:*** $O((n+2)log_2(k))$

##### Performance

We run the ***multithread*** program with varying array sizes and plot their runtimes against the number of array elements. This gives us a clear understanding of the implementations performance metric, as the array size changes. We can also verify the results of our time complexity analysis from this graph.

| Number of Array Elements | Program Runtime (seconds) |
| ------------------------ | ------------------------- |
| 10000                    | 0.000887264               |
| 20000                    | 0.000921152               |
| 30000                    | 0.000948672               |
| 40000                    | 0.00107251                |
| 50000                    | 0.00121709                |
| 60000                    | 0.00130397                |
| 70000                    | 0.00141578                |
| 80000                    | 0.001508                  |
| 90000                    | 0.00159005                |
| 100000                   | 0.0016505                 |
| 200000                   | 0.00273987                |
| 300000                   | 0.00395002                |
| 400000                   | 0.00483267                |
| 500000                   | 0.00569779                |
| 600000                   | 0.00824051                |
| 700000                   | 0.00910086                |
| 800000                   | 0.00999718                |
| 900000                   | 0.0109808                 |
| 1000000                  | 0.0120317                 |
| 2000000                  | 0.0221881                 |
| 3000000                  | 0.0336555                 |

The corresponding graph for the performance measurement is as follows:

<div align="center">
<img src="Multi-Thread_Performance.png" />
</div>

##### Explanation

This implementation shows better performance for larger array sizes, as there are multiple kernel calls, it is not the best possible outcome by any means, but for the specific requirement of sorting 32-bit integers, this algorithm performs much better than it's alternatives. We can also see a small bump in the performance plot, which is caused due to the fact that we are only using the only the maximum number of bits to determine how many keys our radix sort algorithm will use. This causes the number of keys to vary for larger array sizes. 

Of course, this implementation achieves this high-performance by utilizing a big chunk of on device global memory (VRAM). A GPU-utilization measurement using NVIDIA's profiling tools shows us that the Thrust-STL implementation is far more memory efficient in utilizing the device VRAM (around a constant 46-48MiB usage for any load surpassing a set threshold). This optimization is sacrificed in our radix sort alogorithm to achieve the maximum possible sorting speed.

It should also be noted that since we used the array size as a parameter in generating random numbers, so the number of keys will vary in our implementation. This gives us exceptional performance in this case, but might perform badly in a more general case situation. On the other hand, the unoptimized (compliling without enabling the '-O3' flag) thrust STL sorting algorithm's performance becomes worse than the similiarly unoptimized (compiled without enabling the '-O3' flag) multithread algorithm. But the situation immediately reverses upon enabling the compiler flag to optimize both programs - causing the thrust STL's implementation to become much faster than our multithread algorithm which doesn't show much improvement. However, this behavior was an expected one, since this one was already programmed to be as efficient as possible where the NVCC compiler couldn't make any drastic changes.
