#include <iostream>

#include <inttypes.h>
//#include <cuda_device_runtime_api.h>

#include "check_cuda_errors.h"
#include "myrand.h"

// returns a valid result for lane 0 only!
template<typename T>
__device__
unsigned int largestOfWarp(const T* values)
{
    __shared__ int lanes[1024];
    __shared__   T values_shared[1024];

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
T largestOfBlock(const T* values, int value_count)
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
		     /*output*/ int* __restrict__ thread_id)
{
    *thread_id = largestOfBlock(values, value_count);
}

int main()
{
    constexpr size_t N = (256 + 128) * 1024 * 1024;
    auto n = N;

    MyRand myrand;

    float *h_random_array, *d_random_array = nullptr;
    // try to allocate GPU memory first, GPU memory is usually smaller
    checkCudaErrors(
      cudaMalloc(&d_random_array, N * sizeof(*d_random_array))
    );

    int *d_block_map = nullptr;
    checkCudaErrors(
      cudaMalloc(&d_block_map, N/1024 * sizeof(*d_block_map))
    );
    float *d_block_winner = nullptr;
    checkCudaErrors(
      cudaMalloc(&d_block_winner, N/1024 * sizeof(*d_block_winner))
    );


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
    int *d_largest_thread = nullptr;
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

    for (; n >= 1024; n /= 1024) {
        // CUDA kernel with map
    }
    // last CUDA kernel with map

#if 1
    checkCudaErrors(cudaFree(d_largest));
    checkCudaErrors(cudaFree(d_largest_thread));
#endif

    checkCudaErrors(cudaFree(d_block_map));
    checkCudaErrors(cudaFree(d_block_winner));






    checkCudaErrors(cudaFree(d_random_array));
    checkCudaErrors(cudaFreeHost(h_random_array));


}
