import matplotlib.pyplot as plt

plt.figure("Single-Thread Performance")

num_elements_in_array = [x for x in range(10000, 510000, 10000)]
runtimes = [
0.042291,
0.105126,
0.137776,
0.164461,
0.202283,
0.242643,
0.284424,
0.327846,
0.371113,
0.415492,
0.54932,
0.601325,
0.654019,
0.711061,
0.762386,
0.813772,
0.86528,
0.916611,
0.968038,
1.02009,
1.07104,
1.12265,
1.17341,
1.22502,
1.27639,
1.32758,
1.37875,
1.43041,
1.48171,
1.53311,
1.58432,
1.63539,
1.6872,
1.73847,
1.78948,
1.84105,
1.89222,
1.94358,
1.99481,
2.04631,
2.09742,
2.14875,
2.19972,
2.25124,
2.3029,
2.35388,
2.40518,
2.45648,
2.50774,
2.55906
]

plt.plot(num_elements_in_array, runtimes, "-^")
plt.title("Performance Plot For Single-Thread Digit-wise Radix Sort")
plt.xlabel("Number of elements in the array-->")
plt.ylabel("Runtime (in seconds)-->")
# plt.xscale('log')
plt.grid('on')
plt.show()