#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstring>

constexpr int Iterations = 65536;
constexpr int VecSize = 32;
constexpr int BlockSize = 1024;
constexpr int InnerLoop = 3;

template <typename T, typename U, class Op>
__device__ void kernel_ex2(Op op, void* out) {
  T vv[VecSize]{};
  union {
    U srcs[VecSize];
    T dsts[VecSize];
  } u;
#pragma unroll 1
  for (int i = 0; i < Iterations; i++) {
    std::memcpy(u.srcs, vv, sizeof(vv));
    #pragma unroll
    for (int k = 0; k < InnerLoop; k++) {
      #pragma unroll
      for (int j = 0; j < VecSize; j++) {
        op(u.srcs[j]);
      }
    }
    std::memcpy(&u.dsts, &u.srcs, sizeof(vv));
    for (int j = 0; j < VecSize; j++) {
      vv[j] = -u.dsts[j];
    }
  }
  std::memcpy(out, vv, VecSize * sizeof(T));
}

__global__ __launch_bounds__(BlockSize) void kernel_ex2_bf162(void* out) {
  kernel_ex2<__nv_bfloat162, uint32_t>([] __device__ (uint32_t& src) { asm volatile("ex2.approx.ftz.bf16x2 %0, %0;" : "+r"(src) :: "memory"); }, out);
}

__global__ __launch_bounds__(BlockSize) void kernel_ex2_bf16(void* out) {
  kernel_ex2<__nv_bfloat16, uint16_t>([] __device__ (uint16_t& src) { asm volatile("ex2.approx.ftz.bf16 %0, %0;" : "+h"(src) :: "memory"); }, out);
}

__global__ __launch_bounds__(BlockSize) void kernel_ex2_fp32(void* out) {
  kernel_ex2<float, uint32_t>([] __device__ (uint32_t& src) { asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+r"(src) :: "memory"); }, out);
}

int main() {
  int num_sms = 1;
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);

  int64_t num_blocks = num_sms;
  int64_t block_size = BlockSize;

  cudaStream_t stream = 0;
  cudaStreamCreate(&stream);
  cudaEvent_t begin, end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);

  float* out;
  cudaMalloc((void**)&out, VecSize * sizeof(float));

  auto bench = [&](auto name, auto kernel) {
    cudaEventRecord(begin, stream);
    kernel<<<num_blocks, block_size, 0, stream>>>(out);
    cudaEventRecord(end, stream);
    cudaEventSynchronize(end);
    auto err = cudaGetLastError();
    if(err != cudaSuccess) {
      printf("Error: %s\n", cudaGetErrorString(err));
      exit(1);
    }

    float time;
    cudaEventElapsedTime(&time, begin, end);

    float ops = (Iterations * VecSize * InnerLoop * num_blocks * block_size) / (time / 1000);
    float Gops = ops / 1e9;

    printf("%s Time: %f, %f Gop/s\n", name, time, Gops);
  };

  bench("exp2 BF16x2", kernel_ex2_bf162);
  cudaDeviceSynchronize();
  bench("exp2 BF16", kernel_ex2_bf16);
  cudaDeviceSynchronize();
  bench("exp2 FP32", kernel_ex2_fp32);
  cudaDeviceSynchronize();

  cudaStreamDestroy(stream);
  return 0;
}
