//******************************************************************************
// Copyright 2020 ThirtySomething
//******************************************************************************
// This file is part of Sieve.
//
// Sieve is free software: you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option)
// any later version.
//
// Sieve is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for
// more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with Sieve. If not, see <http://www.gnu.org/licenses/>.
//******************************************************************************

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>

__global__ void markAsPrimeKernel(char* vecDevice, long long sieveSize, long long prime)
{
    long long elementId = blockIdx.x * blockDim.x + threadIdx.x;
    long long remainder = elementId % prime;

    if ((elementId > prime) && (0 == remainder) && (elementId < sieveSize))
    {
        vecDevice[elementId] = '\1';
    }
}

void markAsPrimeKernelWrapper(thrust::host_vector<char>& vecHost, long long sieveSize, long long prime)
{
    dim3 dimBlockCount = dim3(ceil(sieveSize / 1024.));
    dim3 dimBlockSize = dim3(1024);

    thrust::device_vector<char> vecDevice(vecHost.size(), '\0');
    thrust::copy(vecHost.begin(), vecHost.end(), vecDevice.begin());

    char* d_ptr = thrust::raw_pointer_cast(vecDevice.data());

    markAsPrimeKernel<<<dimBlockCount, dimBlockSize >>>(d_ptr, sieveSize, prime);

    auto rcValue = cudaDeviceSynchronize();

    thrust::copy(vecDevice.begin(), vecDevice.end(), vecHost.begin());
}