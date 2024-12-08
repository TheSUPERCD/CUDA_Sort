#include<iostream>
#include<sstream>
#include<cmath>
#include<thrust/device_vector.h>
#include<thrust/device_ptr.h>

// function to print arrays (with an optional success/error log-message)
void printFunc(int *array, int arr_size, const char *log_message=""){
    std::cout<< log_message <<"\n[";
    for(int i=0; i<arr_size-1; i++){
        std::cout<< array[i] <<", ";
    }
	std::cout<< array[arr_size-1] << "], size=" << arr_size <<std::endl;
}

// kernel function to create a copy of the given array, while extracting the bit at a given position (marked by the input variable `bit_position`) - all in GPU
__global__ void bitextract_arraycopy(int *array, int *array_copy, int arr_size, int bit_position, int *bin_array){
	int gid = blockDim.x * blockIdx.x + threadIdx.x;			// calculate global ID of the GPU thread
	if(gid < arr_size){
		bin_array[gid] = (array[gid]>>bit_position) & 1;		// bit extraction
		array_copy[gid] = array[gid];							// array copying
	}
}

// kernel function to rearrange the given array using the prefix scan array - in GPU
__global__ void array_sort_shuffled(int *array, int *array_copy, int *bin_array, int *prefix_scan, int arr_size){
	int gid = blockDim.x * blockIdx.x + threadIdx.x;			// calculate global ID of the GPU thread
	int tot_false = arr_size - prefix_scan[arr_size-1];			// total number of false bits in the binary array = array size - number of true bits in the binary array
	if(gid < arr_size){
		if(bin_array[gid])
			array[prefix_scan[gid] -1 + tot_false] = array_copy[gid];
		else
            array[gid - prefix_scan[gid]] = array_copy[gid];
	}
}

// function to take in a given array and sort it using the radix-sort algorithm in GPU
void radix_sort_GPU(int *array, int arr_size, int *array_copy){
	int *bitarray;												// intialize the array for the storing binary bits
	int *prefix_scan;											// intialize the array for the storing the number of true bits before that index position
	
	cudaMalloc((void **)&bitarray, arr_size*sizeof(int));		// allocate memory in gpu
	cudaMalloc((void **)&prefix_scan, arr_size*sizeof(int));	// allocate memory in gpu

	thrust::device_ptr<int> binary_ptr(bitarray);				// convert the pointer into device pointer, to be used in the thrust::inclusive_scan function
	thrust::device_ptr<int> pre_scan_ptr(prefix_scan);			// convert the pointer into device pointer, to be used in the thrust::inclusive_scan function

	int max_bits = (int)log2(arr_size) + 1; 					// maximum number of bits we need to consider for radix sorting
	
	// carry out sorting for each bit in the data-type being sorted (32-bit for `int`)
	for(int i=0; i<max_bits; i++){
		bitextract_arraycopy<<<(arr_size/1024)+1, 1024>>>(array, array_copy, arr_size, i, bitarray);
		thrust::inclusive_scan(binary_ptr, binary_ptr+arr_size, pre_scan_ptr);
		array_sort_shuffled<<<(arr_size/1024)+1, 1024>>>(array, array_copy, bitarray, prefix_scan, arr_size);
	}
	cudaDeviceSynchronize();									// wait for the calculations to finish
	cudaFree(bitarray);											// free VRAM memory
	cudaFree(prefix_scan);										// free VRAM memory
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
	cudaMalloc((void **)&device_array_copy, size*sizeof(int));						
	
	// copy the array values from host to GPU-device
	cudaMemcpy(device_array, array, size*sizeof(int), cudaMemcpyHostToDevice);
	
	// perform massively parallel bitwise radix-sort
	radix_sort_GPU(device_array, size, device_array_copy);
    
	// copy the sorted array back from GPU-device to host memory
	cudaMemcpy(array, device_array, size*sizeof(int), cudaMemcpyDeviceToHost);

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
	cudaFree(device_array_copy);

  std::cerr << "Total time in seconds: "<< (timeTotal / 1000.0) << std::endl;
  
  if(printSorted){
      printFunc(array, size, "\nThe sorted array is:");
  }
  
  free(array);
  return 0;
}
