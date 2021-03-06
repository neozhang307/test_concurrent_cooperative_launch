#include <stdio.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void test0()
{
  cg::grid_group gg = cg::this_grid();
  for(int j=0; j<100000; j++)
  {
    for(int i=0; i<100000; i++)
      asm volatile("nanosleep.u32 1000;");
    printf("execute0\n");
    gg.sync();
  }
  printf("finish0\n");
}
__global__ void test1()
{
  for(int j=0; j<100000; j++)
  {
    for(int i=0; i<100000; i++)
      asm volatile("nanosleep.u32 1000;");
    printf("execute1\n");
  }
  printf("finish1\n");
}

/*
compile:
  nvcc -arch sm_80 -rdc=true cooperative_test.cu
  nvcc -arch sm_80 -rdc=true -DTYPE1 cooperative_test.cu
if the cmd recurrently print "execute0" and "execute1". it means both kernels are concurrently running. 
*/
int main(int argc, char const *argv[])
{
  /* code */
  cudaStream_t stream0;
  cudaStreamCreate( &stream0 ); 
  cudaStream_t stream1;
  cudaStreamCreate( &stream1 ); 
// 2 cooperative
#ifndef TYPE1
  cudaLaunchCooperativeKernel((void*)test0, 1, 1, NULL, 0,stream0);//<-Persistent Kernel Relies on it
  cudaLaunchCooperativeKernel((void*)test1, 1, 1, NULL, 0,stream1);//<-Persistent Kernel Relies on it
#else
//cooperative & traditinal
  cudaLaunchCooperativeKernel((void*)test0, 1, 1, NULL, 0,stream0);//<-Persistent Kernel Relies on it
  test1<<<1,1,0,stream1>>>();
#endif
  cudaDeviceSynchronize();
  cudaStreamDestroy(stream0);
  cudaStreamDestroy(stream1);
  return 0;
}