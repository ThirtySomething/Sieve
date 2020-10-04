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

__global__ void markAsPrimeKernel(long long prime, long long* gpuStorage)
{
    long long elementId = blockIdx.x * blockDim.x + threadIdx.x;
    long long remainder = elementId % prime;

    if ((elementId > prime) && (0 == remainder))
    {
        long long quotientInternal = elementId / 64LL;
        long long remainderInternal = elementId % 64LL;

        long long element = gpuStorage[quotientInternal];
        element |= 1LL << remainderInternal;
        gpuStorage[quotientInternal] = element;
    }
}

void markAsPrimeKernelWrapper(long long sieveSize, long long prime, long long* gpuStorage)
{
    dim3 dimsieveSize = dim3(sieveSize);
    markAsPrimeKernel <<<1, dimsieveSize >>> (prime, gpuStorage);
}

