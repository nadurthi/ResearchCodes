#include <cuda_runtime.h>
#include <iostream>
int main(){
int nDevices;
cudaGetDeviceCount(&nDevices);
std::cout << nDevices << std::endl;
for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    std::cout << prop.major << "." << prop.minor << std::endl;
}
return 0;
}
