#include <iostream>

#include <inttypes.h>
//#include <cuda_device_runtime_api.h>

#include "check_cuda_errors.h"
#include "myrand.h"

// fixed BLOCK_SIZE to try out various values during optimization
constexpr int BLOCK_SIZE = 1024;

// returns a valid result for lane 0 only!
template<typename T>
__device__
unsigned int largestOfWarp(const T* values)
{
    __shared__ int lanes[BLOCK_SIZE];
    __shared__   T values_shared[BLOCK_SIZE];

    int  tid = threadIdx.x;
    //int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    lanes[tid] = tid;
    values_shared[tid] = values[tid];

#pragma unroll
    for (int i = 32 / 2; i > 0; i /= 2) {
        T  this_lane_value = values_shared[lanes[tid]];
        T other_lane_value = __shfl_down_sync(0xffffffff, this_lane_value, i);
#if 0
        if (other_lane_values > this_lane_value)
            lanes[tid] = lanes[tid + i];
        __syncthreads();
#else
        // a bit faster this way, despite of bigger code and multiplication
        // and sometime unnecessary shared mem op
        lanes[tid] = lanes[tid + (other_lane_value > this_lane_value) * i];
#endif
    }
    // return the thread id of the biggest value of this warp (lane 0 surely has it)
    return lanes[warp * 32]; /*** valid only if called from lane 0 of a warp ***/
}

// must be called from all active warps of a block, returns a
// valid result for lane 0 only (including thread 0)
// !!! value_count must be positive and divisible by 32 !!!
template<typename T>
__device__
unsigned int largestOfBlock(const T* values, int value_count)
{
    static __shared__ int largest_threads[32];
    static __shared__   T largest_values[32];

    int tid = threadIdx.x;

    // if value_count is less than 1024, some places left untouched in shared memory,
    // as a precaution, we init the whole thing with 0s (0s are not big)
    // TODO: is it necessary, or can I assume that 'static' ensures 0s?
    if (value_count < 1024 && tid < 32) { largest_values[tid] = 0; }

    // find the largest of this warp
    int winner_thread_of_this_warp = largestOfWarp(values);
    int lane = tid % 32;
    if (lane == 0) {
        int warp = tid / 32;
        largest_threads[warp] = winner_thread_of_this_warp;
        // note: here only lane 0 has the correct winnerLane anyway
        largest_values[warp] = values[winner_thread_of_this_warp];
    }
    __syncthreads();
    int winner_thread = 0;
    if (tid < 32)
    {
        // find the biggests of biggest numbers provided by each warp
        int largest_warp = largestOfWarp(largest_values);
        // winner thread (0-1023) is: biggest lane (0-31) in the biggest warp (0-31)
	// if thread number is less than 1024, lane and warp numbers shrinks accordingly
        winner_thread = largest_threads[largest_warp];
    }
    return winner_thread; /*** valid only if called from lane 0 ***/
}

template<typename T>
__global__
void threadBlockTest(/*input*/  const T* __restrict__ values, int value_count,
		     /*output*/ unsigned int* __restrict__ thread_id)
{
    *thread_id = largestOfBlock(values, value_count);
}

// we need a separate first kernel to initialize block_map
// (otherwise, we would need a huge array with 0,1,2,...,N values)
template<typename T>
__global__
void blocksArgmaxFirst(/*input*/       const T* __restrict__ values,
                       /*output*/ unsigned int* __restrict__ block_map,
                                             T* __restrict__ block_max)
{
    int value_count = blockDim.x;
    int i_out = blockIdx.x;
    int thread_index_offset = i_out * value_count;
    values += thread_index_offset;
    auto largest_thread = largestOfBlock(values, value_count);
    // store one winner thread id and max value per thread block
    if (threadIdx.x == 0) {
        block_max[i_out] = values[largest_thread];
        // in the first step, mapping to the original array is easy
	block_map[i_out] = largest_thread + thread_index_offset;
    }
}

template <typename T>
__global__
void blocksArgmax(/* inputs, both are the size of N */
                   const unsigned int* __restrict__ prev_block_map,
                              const T* __restrict__ prev_block_max,
                  /* output, both are the size of N/value_count */
                         unsigned int* __restrict__ block_map,
                                    T* __restrict__ block_max
                 )
{
    int value_count = blockDim.x; // value count = thread count in this block
    int       i_out = blockIdx.x; // output index (block id)
    unsigned int thread_index_offset = i_out * value_count;

    // select from prev maxes for this thread block
    prev_block_max += thread_index_offset;

    auto largest_thread = largestOfBlock(prev_block_max, value_count);
    // store one winner thread index and max value per thread block
    // note: winner thread index needs mapping as processed arrays shrink during
    // the iteration (we still need the indices from the original huge array)
    if (threadIdx.x == 0) {
        // max value is easy
        block_max[i_out] = prev_block_max[largest_thread];

        // the thread index must be an index in the original huge array
        prev_block_map  += thread_index_offset;
        largest_thread   = prev_block_map[largest_thread]; // mapping here
        block_map[i_out] = largest_thread;
    }
}

int main()
{
    constexpr size_t N = (256 + 128) * 1024 * 1024;
    //constexpr size_t N = (32) * 1024 * 1024;
    auto n = N;


    MyRand myrand;

    float *h_random_array, *d_random_array = nullptr;
    // try to allocate GPU memory first, GPU memory is usually smaller
    checkCudaErrors(
      cudaMalloc(&d_random_array, N * sizeof(*d_random_array))
    );

    float *d_block_max = nullptr;
    checkCudaErrors(
      cudaMalloc(&d_block_max, N/BLOCK_SIZE * 2 * sizeof(*d_block_max))
    );
    unsigned int *d_block_map = nullptr;
    checkCudaErrors(
      cudaMalloc(&d_block_map, N/BLOCK_SIZE * 2 * sizeof(*d_block_map))
    );
    size_t block_map_selector = 0;

    // page locked CPU memory for later chunck by chunk async memcopy (TODO)
    checkCudaErrors(
      cudaMallocHost(&h_random_array, N * sizeof(*h_random_array))
    );

    for (auto i = 0; i < N; ++i) {
        // TODO: if it is slow, use CUDA kernel (note: then deviceToHost copy needed)
        h_random_array[i] = static_cast<float>(myrand()) / static_cast<unsigned int>(-1);
	h_random_array[i] *= 1'000'000;
	//std::cout << h_random_array[i] << "\n";
    }

    // TODO: cut host array into pieces, do async memcopy chunks on other stream
    checkCudaErrors(
      cudaMemcpy(d_random_array, h_random_array, N * sizeof(*h_random_array),
	         cudaMemcpyHostToDevice)
    );

#if 1
    constexpr int experiment_count = 100;

    // now this test code is the only thing that is really used
    unsigned int *d_largest_thread = nullptr;
    float *d_largest = nullptr;

    checkCudaErrors(
      cudaMalloc(&d_largest_thread, sizeof(*d_largest_thread))
    );
    checkCudaErrors(
      cudaMalloc(&d_largest, sizeof(*d_largest))
    );

    // it runs for testN values only, instead of N
    int testN = 256+128;

    for (int ex = 0; ex < experiment_count; ++ex)
    {
        int offset = myrand() % (N - testN - 1);

        threadBlockTest<<<1, testN>>>(d_random_array + offset, testN,
                                      d_largest_thread);
        int h_largest_thread = -1;

        checkCudaErrors(
          cudaMemcpy(&h_largest_thread, d_largest_thread, sizeof(h_largest_thread),
                     cudaMemcpyDeviceToHost)
        );
        h_largest_thread += offset;
        std::cout << "GPU found largest value on index " << h_largest_thread
                  << ", its value is " << h_random_array[h_largest_thread] << "\n";

        // let's check the result on the CPU
        float largest = 0;
          int largest_index = 0;
        for (int i = offset; i < offset + testN; ++i) {
            if (h_random_array[i] > largest) {
                largest = h_random_array[i];
                largest_index = i;
            }
        }

        std::cout << "CPU found largest value on index " << largest_index
                  << ", its value is " << largest << "\n";
        std::cout << (largest_index == h_largest_thread? "CHECKED.\n" : "ERROR!\n");
        std::cout << (largest_index == h_largest_thread?
	            "---------------------------------------------------------------\n"
	          : "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    }
#endif
#if 1
    // first largest value per block finding step, also initializes index map
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int numThreads = n > BLOCK_SIZE ? BLOCK_SIZE : n;
    blocksArgmaxFirst<<< numBlocks, numThreads >>>(
		    /* input  */d_random_array,
		    /* output */d_block_map, d_block_max
		    );
#if 1
    // remaining steps with input and output index map with
    // corresponding max values
    while ((n /= BLOCK_SIZE) && n != 1) {
        int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
	int numThreads = n > BLOCK_SIZE ? BLOCK_SIZE : n;
        std::cout << n << " -> <<< " << numBlocks << ", "
		  << numThreads << " >>>\n";
	blocksArgmax<<< numBlocks, numThreads >>>(
			/* input */
                        d_block_map + block_map_selector,
                        d_block_max + block_map_selector,
                        /* output */
                        d_block_map + N/BLOCK_SIZE - block_map_selector,
			d_block_max + N/BLOCK_SIZE - block_map_selector
                       );
        block_map_selector = N/BLOCK_SIZE - block_map_selector;
    }

    unsigned int h_largest_index = 0;
    checkCudaErrors(
      cudaMemcpy(&h_largest_index, d_block_map + block_map_selector,
	           sizeof(h_largest_index), cudaMemcpyDeviceToHost)
    );

    float h_largest = 0;
    checkCudaErrors(
      cudaMemcpy(&h_largest, d_block_max + block_map_selector,
	           sizeof(h_largest), cudaMemcpyDeviceToHost)
    );

    std::cout << "GPU found the overall largest value " << h_largest
              << " on index " << h_largest_index << "\n";

    float largest = 0;
      int largest_index = 0;
    for (unsigned int i = 0; i < N; ++i) {
        if (h_random_array[i] > largest) {
            largest = h_random_array[i];
            largest_index = i;
        }
    }

    std::cout << "CPU found the overall largest value " << largest
              << " on index " << largest_index << "\n";


#endif
#endif
#if 1
    checkCudaErrors(cudaFree(d_largest));
    checkCudaErrors(cudaFree(d_largest_thread));
#endif

    checkCudaErrors(cudaFree(d_block_max));
    checkCudaErrors(cudaFree(d_block_map));






    checkCudaErrors(cudaFree(d_random_array));
    checkCudaErrors(cudaFreeHost(h_random_array));


}
