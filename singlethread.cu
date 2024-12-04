#include<iostream>
#include<sstream>
#include<cmath>
#include<thrust/device_vector.h>

// function to print arrays (with an optional success/error log-message)
void printFunc(int *array, int arr_size, const char *log_message=""){
    std::cout<< log_message <<"\n[";
    for(int i=0; i<arr_size-1; i++){
        std::cout<< array[i] <<", ";
    }
	std::cout<< array[arr_size-1] << "], size=" << arr_size <<std::endl;
}

// kernel function to perform radix sort using a single GPU thread
__global__ void radixSort(int *array, int *array_copy, int arr_size, unsigned long max_decimal){
    
    // carry out radix sort for each decimal digit position (from 10^0 to 10^x where 10^x is less than `max_decimal`)
    for(int digit_multiplier=1; digit_multiplier<max_decimal; digit_multiplier*=10){
        
        // counting total number of a certain digit (from 0-9)
        int digit_position_counter[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        for(int i=0; i<arr_size; i++)
            digit_position_counter[(array[i] / digit_multiplier) % 10]++;
        
        // prefix scan
        for(int i=1; i<10; i++)
            digit_position_counter[i] += digit_position_counter[i-1];
        
        // rearrange the numbers in `array` and putting them in `array_copy` to preserve their original order in `array`
        for(int i=arr_size-1; i>=0; i--){
            array_copy[digit_position_counter[(array[i] / digit_multiplier) % 10] - 1] = array[i];
            digit_position_counter[(array[i] / digit_multiplier) % 10]--;
        }
        
        // copying `array_copy` to `array` - since sorting is over for this digit position, we don't need the original order anymore
        for(int i=0; i<arr_size; i++)
            array[i] = array_copy[i];
    }
}

/**********************************************************
***********************************************************
error checking stufff
***********************************************************
**********************************************************/

// Enable this for error checking
// #define CUDA_CHECK_ERROR
#define CudaSafeCall( err )     __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()        __cudaCheckError( __FILE__, __LINE__ )


inline void __cudaSafeCall(cudaError err,const char *file, const int line)
{
	#ifdef CUDA_CHECK_ERROR
	#pragma warning( push )
	#pragma warning( disable: 4127 )    // Prevent warning on do-while(0)
	
    do
		{
			if (cudaSuccess != err)
			{
				fprintf(stderr,"cudaSafeCall() failed at %s:%i : %s\n",file, line, cudaGetErrorString(err));
				exit(-1);
			}
		} while (0);
	#pragma warning( pop )
	#endif  // CUDA_CHECK_ERROR
	return;
}

inline void __cudaCheckError(const char *file, const int line)
{
	#ifdef CUDA_CHECK_ERROR
	#pragma warning( push )
	#pragma warning( disable: 4127 )    // Prevent warning on do-while(0);

	do
	{
		cudaError_t err = cudaGetLastError();
		if (cudaSuccess != err)
		{
			fprintf(stderr,"cudaCheckError() failed at %s:%i : %s.\n", file, line, cudaGetErrorString(err));
			exit(-1);
		}
        // More careful checking. However, this will affect performance.
        // Comment if not needed.
		err = cudaThreadSynchronize();
		if (cudaSuccess != err)
		{
			fprintf(stderr,"cudaCheckError() with sync failed at %s:%i : %s.\n", file, line, cudaGetErrorString(err));
			exit(-1);
		}
	} while (0);
	
	#pragma warning( pop )
	#endif    // CUDA_CHECK_ERROR
	return;
}
/***************************************************************
****************************************************************
end of error checking stuff
****************************************************************
***************************************************************/

// function takes an array pointer and the size of the array, and
// allocates and intializes the array to a bunch of random numbers
int *makeRandArray(const int size, const int seed){
	srand(seed);
    int *array;
    array = (int *)malloc(size*sizeof(int));
	for (int i = 0; i < size; i++){
		array[i] = rand() % size;
	}
	return array;
}

int main(int argc, char* argv[]){
    int *array;
    int size, seed; // values for the size of the array
    bool printSorted = false;
    // and the seed for generating
    // random numbers
    // check the command line args
    if (argc < 4){
    	std::cerr << "usage: "
    		<< argv[0]
    		<< " [amount of random nums to generate] [seed value for rand]"
    		<< " [1 to print sorted array, 0 otherwise]"
    		<< std::endl;
    	exit(-1);
    }
    
    {
        std::stringstream ss1(argv[1]);
        ss1 >> size;
    }
    
    {
        std::stringstream ss1(argv[2]);
        ss1 >> seed;
    }

    int sortPrint;
    std::stringstream ss1(argv[3]);
    ss1 >> sortPrint;

    if (sortPrint == 1){
        printSorted = true;
    }   
    
    // creating an array of random elements
    array = makeRandArray(size, seed);

    /***********************************
    create a cuda timer to time execution
    ***********************************/
    cudaEvent_t startTotal, stopTotal;
    float timeTotal;
    cudaEventCreate(&startTotal);
    cudaEventCreate(&stopTotal);
    cudaEventRecord(startTotal, 0);
    /***********************************
    end of cuda timer creation
    ***********************************/ 

    /////////////////////////////////////////////////////////////////////
    
    // intializing the device array pointers
    int *device_array;
    int *device_array_copy;

    // allocating memory for the device arrays on GPU
    cudaMalloc((void **)&device_array, size*sizeof(int));
    cudaMalloc((void **)&device_array_copy, (size+1)*sizeof(int));

    // copy the array values from host to GPU-device
    cudaMemcpy(device_array, array, size*sizeof(int), cudaMemcpyHostToDevice);
    
    // perform radix sort using a single GPU-thread
    unsigned long max_decimal = pow(10, ceil(log10(size)));
    radixSort<<<1,1>>>(device_array, device_array_copy, size, max_decimal);
    
    // copy the sorted array back from GPU-device to host memory
    cudaMemcpy(array, device_array, size*sizeof(int), cudaMemcpyDeviceToHost);

    /////////////////////////////////////////////////////////////////////
	/* 	You need to implement your kernel as a function at the top of this file.
	*	Here you must
	*
	*	1) allocate device memory
	*	2) set up the grid and block sizes
	*	3) call your kenrnel
	*	4) get the result back from the GPU
	*
	*	to use the error checking code, wrap any cudamalloc functions as follows:
	*	CudaSafeCall( cudaMalloc( &pointer_to_a_device_pointer,
	*	length_of_array * sizeof( int ) ) );
	*	Also, place the following function call immediately after you call your kernel
	*	( or after any other cuda call that you think might be causing an error )
	*	CudaCheckError();
	*/
    
    
    /***********************************
    stop and destroy the cuda timer
    ***********************************/
    cudaEventRecord(stopTotal, 0);
    cudaEventSynchronize(stopTotal);
    cudaEventElapsedTime(&timeTotal, startTotal, stopTotal);
    cudaEventDestroy(startTotal);
    cudaEventDestroy(stopTotal);
    /***********************************
    end of cuda timer destruction
    ***********************************/ 
    
    cudaFree(device_array);

    std::cerr << "Total time in seconds: "
    	<< timeTotal / 1000.0 << std::endl;
    if (printSorted){
        printFunc(array, size, "\nThe sorted array is:");

    }
    free(array);
    return 0;
}
