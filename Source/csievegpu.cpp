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

#include "CSieveGPU.h"
#include <iostream>
#include <stdlib.h>
#include <sstream>
#include <thread>
#include <windows.h>

// *****************************************************************************
// *****************************************************************************
extern void markAsPrimeKernelWrapper(thrust::host_vector<char>& vecHost, long long sieveSize, long long prime);

// *****************************************************************************
// *****************************************************************************
namespace net
{
    // *****************************************************************************
    // *****************************************************************************
    namespace derpaul
    {
        // *****************************************************************************
        // *****************************************************************************
        namespace sieve
        {
            // *****************************************************************************
            // Constants
            // *****************************************************************************
            const long long CSieveGPU::DEFAULT_SIEVE_SIZE = 10000LL;
            const unsigned int CSieveGPU::m_numberOfCores = std::thread::hardware_concurrency();

            // *****************************************************************************
            // *****************************************************************************
            CSieveGPU::CSieveGPU(long long sieveSize) : m_sieveSize(sieveSize),
                                                  m_latestPrime(1LL),
                                                  m_stop_work(false),
                                                  m_storage(sieveSize)
            {
                initStorage();
            }

            // *****************************************************************************
            // *****************************************************************************
            CSieveGPU::~CSieveGPU(void)
            {
            }

            // *****************************************************************************
            // *****************************************************************************
            void CSieveGPU::dataLoad(std::string filename)
            {
                auto [latestPrime, sieveSize] = m_storage.dataLoad(filename);
                m_latestPrime = latestPrime;
                m_sieveSize = sieveSize;
            }

            // *****************************************************************************
            // *****************************************************************************
            void CSieveGPU::dataSave(std::string filename)
            {
                m_storage.dataSave(filename, m_latestPrime, m_sieveSize);
            }

            // *****************************************************************************
            // *****************************************************************************
            void CSieveGPU::exportPrimes(std::string filename)
            {
                m_storage.exportPrimes(filename, m_latestPrime);
            }

            // *****************************************************************************
            // *****************************************************************************
            long long CSieveGPU::getLatestPrime(void)
            {
                return m_latestPrime;
            }

            // *****************************************************************************
            // *****************************************************************************
            long long CSieveGPU::getSieveSize(void)
            {
                return m_sieveSize;
            }

            // *****************************************************************************
            // *****************************************************************************
            void CSieveGPU::interruptSieving(void)
            {
                m_stop_work = true;
            }

            // *****************************************************************************
            // *****************************************************************************
            void CSieveGPU::sievePrimes(std::function<void(long long)> updatePrime)
            {
                m_stop_work = false;

                // https://docs.nvidia.com/cuda/thrust/index.html

                while (!m_stop_work && (m_latestPrime < m_sieveSize))
                {
                    long long primeTemp = m_storage.findNextPrime(m_latestPrime);
                    updatePrime(primeTemp);
                    markAsPrimeKernelWrapper(m_storage.m_storage, m_sieveSize, primeTemp);
                    if (!m_stop_work)
                    {
                        m_latestPrime = primeTemp;
                    }
                }

/**
                char* gpuStorage;
                size_t storageSize = m_storage.getStorageSize() * sizeof(char);
                auto rcValueMal = cudaMalloc((void**)&gpuStorage, storageSize);
                char* dataArray = m_storage.getStoragePointer();
                dataArray[0] = 'E';
                auto rcValueCpy = cudaMemcpy(gpuStorage, dataArray, storageSize, cudaMemcpyHostToDevice);

                while (!m_stop_work && (m_latestPrime < m_sieveSize))
                {
                    long long primeTemp = m_storage.findNextPrime(m_latestPrime);
                    updatePrime(primeTemp);
                    markAsPrimeKernelWrapper(m_sieveSize, primeTemp, gpuStorage);
                    if (!m_stop_work)
                    {
                        m_latestPrime = primeTemp;
                    }
                    rcValueCpy = cudaMemcpy(dataArray, gpuStorage, storageSize, cudaMemcpyDeviceToHost);
                }

                auto rcValueFree = cudaFree(gpuStorage);

                thrust::device_vector<char> gpuStorage(m_storage.getStorageSize(), net::derpaul::sieve::CDataStorage::m_unset);
                thrust::host_vector<char> cpuStorage = m_storage.getStorageReference();
                // thrust::copy(cpuStorage.begin(), cpuStorage.end(), gpuStorage.begin());
**/
            }

            // *****************************************************************************
            // *****************************************************************************
            void CSieveGPU::initStorage(void)
            {
                m_storage.clear();
                m_storage.markNumberAsNotPrime(0LL);
                m_storage.markNumberAsNotPrime(1LL);
                m_latestPrime = 1LL;
            }
        } // namespace sieve
    }     // namespace derpaul
} // namespace net
