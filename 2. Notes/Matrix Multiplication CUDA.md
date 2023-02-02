本文基于ECA-GPU实验指导。
# 矩阵乘法问题
## 几种优化技术
1.  Tiling 分块
2.  Memory coalescing 内存合并
3.  Avoiding memory bank conflicts  避免内存冲突
4.  Computation Optimization.  计算优化
5.  Loop unrolling 循环展开
6.  Prefetching 预取
## 基础问题
- CPU上简单实现：
```C
void main(){
  define A, B, C
  for i = 0 to M do
    for j = 0 to N do
      /* compute element C(i,j) */
      for k = 0 to K do
        C(i,j) < = C(i,j) + A(i,k) * B(k,j)
      end
    end
  end
}
```
![[Pasted image 20230101160800.png]]
- GPU上实现
```C
/* Codes running on CPU */
void main(){

    define A_cpu, B_cpu, C_cpu in the CPU memory
    define A_gpu, B_gpu, C_gpu in the GPU memory

    memcopy A_cpu to A_gpu
    memcopy B_cpu to B_gpu

    dim3 dimBlock(16, 16)
    dim3 dimGrid(N/dimBlock.x, M/dimBlock.y)
	/*N=M=k=32*/
    matrixMul<<<dimGrid, dimBlock>>>(A_gpu,B_gpu,C_gpu,K)

    memcopy C_gpu to C_cpu

}
```

```C
/* Codes running on GPU */
__global__ void matrixMul(A_gpu,B_gpu,C_gpu,K){

    temp < = 0
	/*blockDim=16*/
    i < = blockIdx.y * blockDim.y + threadIdx.y    // Row i of matrix C
    j < = blockIdx.x * blockDim.x + threadIdx.x    // Column j of matrix C

    for k = 0 to K-1 do
        accu < = accu + A_gpu(i,k) * B_gpu(k,j)
    end

    C_gpu(i,j) < = accu

}
```
![[Pasted image 20230101161144.png]]
# Tiling 分块
![[Pasted image 20230101163826.png]]
通过两次迭代计算$C_{0,0}$
![[Pasted image 20230101163837.png]]
![[Pasted image 20230101163856.png]]
将结果保存到Global Menory中，下次迭代结果在累加进去。
![[Pasted image 20230101163909.png]]
```C
/* Codes running on GPU */

__global__ void matrixMul(A_gpu,B_gpu,C_gpu,K){

    __shared__ float A_tile(blockDim.y, blockDim.x)
    __shared__ float B_tile(blockDim.x, blockDim.y)

    accu < = 0

    /* Accumulate C tile by tile. */

    for tileIdx = 0 to (K/blockDim.x - 1) do

        /* Load one tile of A and one tile of B into shared mem */

        // Row i of matrix A
        i < = blockIdx.y * blockDim.y + threadIdx.y
        // Column j of matrix A
        j < = tileIdx * blockDim.x + threadIdx.x
        // Load A(i,j) to shared mem
        A_tile(threadIdx.y, threadIdx.x) < = A_gpu(i,j)
        // Load B(j,i) to shared mem
        B_tile(threadIdx.x, threadIdx.y) < = B_gpu(j,i) // Global Mem Not coalesced
        // Synchronize before computation
        __sync()

        /* Accumulate one tile of C from tiles of A and B in shared mem */

        for k = 0 to threadDim.x do
            // Accumulate for matrix C
            accu < = accu + A_tile(threadIdx.y,k) * B_tile(k,threadIdx.x)
        end
        // Synchronize
        __sync()

    end

    // Row i of matrix C
    i < = blockIdx.y * blockDim.y + threadIdx.y
    // Column j of matrix C
    j < = blockIdx.x * blockDim.x + threadIdx.x
    // Store accumulated value to C(i,j)
    C_gpu(i,j) < = accu

}
```
# Global Memory Coalescing 内存合并
在C/C++的二维数组中，通常是行优先存储。在上面的分块实现中，相邻线程合并了对矩阵 A 的访问，但没有合并访问矩阵 B。在列优先语言（例如 Fortran）中，问题正好相反。一个明显的解决方案是在将矩阵 B加载到 GPU 内存之前由 CPU 对其进行转置。
```C
/* Codes running on GPU */

__global__ void matrixMul(A_gpu,B_gpu,C_gpu,K){

    __shared__ float A_tile(blockDim.y, blockDim.x)
    __shared__ float B_tile(blockDim.x, blockDim.y)

    accu < = 0

    /* Accumulate C tile by tile. */

    for tileIdx = 0 to (K/blockDim.x - 1) do

        /* Load one tile of A and one tile of B into shared mem */

        // Row i of matrix A
        i < = blockIdx.y * blockDim.y + threadIdx.y
        // Column j of matrix A
        j < = tileIdx * blockDim.x + threadIdx.x
        // Load A(i,j) to shared mem
        A_tile(threadIdx.y, threadIdx.x) < = A_gpu(i,j)
        // Load B(i,j) to shared mem,transpose matrix B
        B_tile(threadIdx.x, threadIdx.y) < = B_gpu(i,j) // Global Mem Coalesced
        // Synchronize before computation
        __sync()

        /* Accumulate one tile of C from tiles of A and B in shared mem */

        for k = 0 to threadDim.x do
            // Accumulate for matrix C    // Shared Mem Bank conflict
            accu < = accu + A_tile(threadIdx.y,k) * B_tile(threadIdx.x,k)
        end
        // Synchronize
        __sync()

    end

    // Row i of matrix C
    i < = blockIdx.y * blockDim.y + threadIdx.y
    // Column j of matrix C
    j < = blockIdx.x * blockDim.x + threadIdx.x
    // Store accumulated value to C(i,j)
    C_gpu(i,j) < = accu

}
```
# Avoiding Shared Memory Bank Conflict
```C
/* Codes running on GPU */

__global__ void matrixMul(A_gpu,B_gpu,C_gpu,K){

    __shared__ float A_tile(blockDim.y, blockDim.x)
    __shared__ float B_tile(blockDim.x, blockDim.y)

    accu < = 0

    /* Accumulate C tile by tile. */

    for tileIdx = 0 to (K/blockDim.x - 1) do

        /* Load one tile of A and one tile of B into shared mem */

        // Row i of matrix A
        i < = blockIdx.y * blockDim.y + threadIdx.y
        // Column j of matrix A
        j < = tileIdx * blockDim.x + threadIdx.x
        // Load A(i,j) to shared mem
        A_tile(threadIdx.y, threadIdx.x) <= A_gpu(i,j)
        // Load B(i,j) to shared mem
        B_tile(threadIdx.y, threadIdx.x) <= B_gpu(i,j) // No Shared Mem Bank conflict
        // Synchronize before computation
        __sync()

        /* Accumulate one tile of C from tiles of A and B in shared mem */

        for k = 0 to threadDim.x do
            // Accumulate for matrix C  // No Shared Mem Bank conflict
            accu < = accu + A_tile(threadIdx.y,k) * B_tile(k,threadIdx.x)
        end
        // Synchronize
        __sync()

    end

    // Row i of matrix C
    i <= blockIdx.y * blockDim.y + threadIdx.y
    // Column j of matrix C
    j <= blockIdx.x * blockDim.x + threadIdx.x
    // Store accumulated value to C(i,j)
    C_gpu(i,j) <= accu

}
```
# Computation Optimization计算优化
内积占用时间，采用外积。在这种情况下，矩阵 A 存储在共享内存中，而矩阵 B 和 C 存储在寄存器中。外积不需要共享矩阵B和矩阵C，因此每个线程在寄存器中只存储B的一个元素和C的tile的一列。
```C
/* CUDA code for inner product */
accu < = accu + A_tile(threadIdx.y,k) * B_tile(k, threadIdx.x)
/* CUDA code for outer product */
/* accu[i] and b are stored in register file */
accu[i] < = accu[i] + A_tile(i) * b
```
![[Pasted image 20230101171413.png]]
![[Pasted image 20230101171428.png]]

# Loop Unrolling循环展开
使用编译指示告诉编译器展开循环。默认情况下，nvcc 将展开内部循环。但它不会展开外循环，除非`#pragma unroll`告诉它。
循环展开有时会对寄存器使用产生副作用，这可能会限制并发线程的数量。但是，循环展开不会增加 matrixMul 示例中的寄存器使用量。
# Prefetching 预取
```C
/* Codes running on GPU */

__global__ void matrixMul(A_gpu,B_gpu,C_gpu,K){

    __shared__ float A_tile0(blockDim.y, blockDim.x)
    __shared__ float A_tile1(blockDim.x, blockDim.y)
	
    float *pointer0 = A_tile0
    float *pointer1 = A_tile1

    fetch one tile of matrix A_gpu to pointer0

    __sync()

    /* Accumulate C tile by tile. */

    for tileIdx = 0 to (K/blockDim.x - 1) do

        prefetch one tile of matrix A_gpu to pointer1

        accumulate C using pointer0

        __sync()

        swap pointer0 and pointer1

    end

    store tile C to global memory

}
```

