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

__global__ void markAsPrimeKernel(char* vecDevice, long long sieveSize, long long prime)
{
    long long elementId = (blockIdx.x * blockDim.x + threadIdx.x) + 2;
    long long idx = elementId * prime;

    if ((idx < sieveSize))
    {
        vecDevice[idx] = '\1';
    }
}

void markAsPrimeKernelWrapper(char* vecDevice, long long sieveSize, long long prime)
{
    dim3 dimBlockCount = dim3(ceil((sieveSize / (double)prime) / 1024.));
    dim3 dimBlockSize = dim3(1024);

    markAsPrimeKernel<<<dimBlockCount, dimBlockSize >>>(vecDevice, sieveSize, prime);

    //auto rcValue = cudaDeviceSynchronize();
}

//long long primeTemp = findNextPrimeKernelWrapper(d_ptr, m_sieveSize, m_latestPrime);