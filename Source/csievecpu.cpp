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

#include "csievecpu.h"
#include <iostream>
#include <stdlib.h>
#include <sstream>
#include <thread>
#include <vector>
#include <windows.h>

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
            const long long CSieveCPU::DEFAULT_SIEVE_SIZE = 10000LL;
            const unsigned int CSieveCPU::m_numberOfCores = std::thread::hardware_concurrency();

            // *****************************************************************************
            // *****************************************************************************
            CSieveCPU::CSieveCPU(long long sieveSize) : m_sieveSize(sieveSize),
                                                  m_latestPrime(1LL),
                                                  m_stop_work(false)
            {
                initStorage();
            }

            // *****************************************************************************
            // *****************************************************************************
            CSieveCPU::~CSieveCPU(void)
            {
            }

            // *****************************************************************************
            // *****************************************************************************
            void CSieveCPU::dataLoad(std::string filename)
            {
                auto [latestPrime, sieveSize] = m_storage.dataLoad(filename);
                m_latestPrime = latestPrime;
                m_sieveSize = sieveSize;
            }

            // *****************************************************************************
            // *****************************************************************************
            void CSieveCPU::dataSave(std::string filename)
            {
                m_storage.dataSave(filename, m_latestPrime, m_sieveSize);
            }

            // *****************************************************************************
            // *****************************************************************************
            void CSieveCPU::exportPrimes(std::string filename)
            {
                m_storage.exportPrimes(filename, m_latestPrime);
            }

            // *****************************************************************************
            // *****************************************************************************
            long long CSieveCPU::getLatestPrime(void)
            {
                return m_latestPrime;
            }

            // *****************************************************************************
            // *****************************************************************************
            long long CSieveCPU::getSieveSize(void)
            {
                return m_sieveSize;
            }

            // *****************************************************************************
            // *****************************************************************************
            void CSieveCPU::interruptSieving(void)
            {
                m_stop_work = true;
            }

            // *****************************************************************************
            // *****************************************************************************
            void CSieveCPU::sievePrimes(std::function<void(long long)> updatePrime)
            {
                long long primeTemp = 0LL;
                m_stop_work = false;

                while (!m_stop_work && (m_latestPrime < m_sieveSize))
                {
                    primeTemp = m_storage.findNextPrime(m_latestPrime);
                    updatePrime(primeTemp);
                    markPrimeMultiples(primeTemp);
                    if (!m_stop_work)
                    {
                        m_latestPrime = primeTemp;
                    }
                }
            }

            // *****************************************************************************
            // *****************************************************************************
            void CSieveCPU::initStorage(void)
            {
                m_storage.clear();
                m_storage.markNumberAsNotPrime(0LL);
                m_storage.markNumberAsNotPrime(1LL);
                m_latestPrime = 1LL;
            }

            // *****************************************************************************
            // *****************************************************************************
            void CSieveCPU::markPrimeMultiples(long long prime)
            {
                std::vector<std::future<void> > markThreads;
                lldiv_t parts = lldiv(m_sieveSize, static_cast<long long>(m_numberOfCores));

                for (long long i = 0; i < m_numberOfCores; i++)
                {
                    if ((parts.quot * (i + 1LL)) < prime)
                    {
                        continue;
                    }
                    auto markSegment = std::async(std::launch::async, &CSieveCPU::markPrimeMultiplesSegment, this, (parts.quot * i), (parts.quot * (i + 1LL)), prime);
                    markThreads.push_back(std::move(markSegment));
                }
                auto markSegment = std::async(std::launch::async, &CSieveCPU::markPrimeMultiplesSegment, this, (parts.quot * m_numberOfCores), m_sieveSize, prime);
                markThreads.push_back(std::move(markSegment));

                for (auto &th : markThreads)
                {
                    th.wait();
                }
            }

            // *****************************************************************************
            // *****************************************************************************
            void CSieveCPU::markPrimeMultiplesSegment(long long segmentStart, long long segmentEnd, long long prime)
            {
                lldiv_t parts = lldiv(segmentStart, prime);

                long long startMark = (parts.quot + 1LL) * prime;
                if (parts.rem == 0LL)
                {
                    startMark = parts.quot * prime;
                }

                startMark = std::max<long long>(startMark, (prime * 2));
                for (long long primeMultiple = startMark; ((!m_stop_work) && (primeMultiple < segmentEnd)); primeMultiple += prime)
                {
                    m_storage.markNumberAsNotPrime(primeMultiple);
                }
            }

        } // namespace sieve
    }     // namespace derpaul
} // namespace net
