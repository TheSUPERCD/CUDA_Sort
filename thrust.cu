#include<iostream>
#include<sstream>
#include<thrust/sort.h>

// function to print arrays (with an optional success/error log-message)
void printFunc(int *array, int arr_size, const char *log_message=""){
    std::cout<< log_message <<"\n[";
    for(int i=0; i<arr_size-1; i++){
        std::cout<< array[i] <<", ";
    }
	std::cout<< array[arr_size-1] << "], size=" << arr_size <<std::endl;
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

    // sorting the array using Thrust STL
    thrust::sort(array, array + size);
    
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
    
    std::cerr << "Total time in seconds: "
    	<< timeTotal / 1000.0 << std::endl;
    if (printSorted){
        printFunc(array, size, "\nThe sorted array is:");

    }
    free(array);
    return 0;
}