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

__global__ void markAsPrimeKernel(thrust::device_ptr<char> vecDevice, long long prime)
{
    long long elementId = blockIdx.x * blockDim.x + threadIdx.x;
    long long remainder = elementId % prime;

    if ((elementId > prime) && (0 == remainder))
    {
        vecDevice[elementId] = '\1';
    }
}

void markAsPrimeKernelWrapper(thrust::host_vector<char>& vecHost, long long prime)
{
    dim3 dimSieveSize = dim3(vecHost.size());

    thrust::device_vector<char> vecDevice(vecHost.size(), '\0');
    thrust::copy(vecHost.begin(), vecHost.end(), vecDevice.begin());

    markAsPrimeKernel<<<1, dimSieveSize>>>(vecDevice.data(), prime);

    auto rcValue = cudaDeviceSynchronize();

    vecHost.clear();
    thrust::copy(vecDevice.begin(), vecDevice.end(), vecHost.begin());
}