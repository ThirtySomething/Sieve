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

#include "csieve.h"
#include <iostream>
#include <windows.h>
#include <thread>

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
            const long long CSieve::DEFAULT_SIEVE_SIZE = 10000LL;

            // *****************************************************************************
            // *****************************************************************************
            CSieve::CSieve(long long sieveSize) : m_sieveSize(sieveSize),
                                                  m_latestPrime(1LL),
                                                  m_stop_work(false)
            {
                initStorage();
            }

            // *****************************************************************************
            // *****************************************************************************
            CSieve::~CSieve(void)
            {
            }

            // *****************************************************************************
            // *****************************************************************************
            void CSieve::dataLoad(std::string filename)
            {
                auto [latestPrime, sieveSize] = m_storage.dataLoad(filename);
                m_latestPrime = latestPrime;
                m_sieveSize = sieveSize;
            }

            // *****************************************************************************
            // *****************************************************************************
            void CSieve::dataSave(std::string filename)
            {
                m_storage.dataSave(filename, m_latestPrime, m_sieveSize);
            }

            // *****************************************************************************
            // *****************************************************************************
            void CSieve::exportPrimes(std::string filename)
            {
                m_storage.exportPrimes(filename, m_latestPrime);
            }

            // *****************************************************************************
            // *****************************************************************************
            long long CSieve::getLatestPrime(void)
            {
                return m_latestPrime;
            }

            // *****************************************************************************
            // *****************************************************************************
            long long CSieve::getSieveSize(void)
            {
                return m_sieveSize;
            }

            // *****************************************************************************
            // *****************************************************************************
            void CSieve::interruptSieving(void)
            {
                m_stop_work = true;
            }

            // *****************************************************************************
            // *****************************************************************************
            void CSieve::sievePrimes(std::function<void(long long)> updatePrime)
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
            void CSieve::initStorage(void)
            {
                m_storage.clear();
                m_storage.markNumberAsNotPrime(0LL);
                m_storage.markNumberAsNotPrime(1LL);
                m_latestPrime = 1LL;
            }

            // *****************************************************************************
            // *****************************************************************************
            void CSieve::markPrimeMultiples(long long prime)
            {
                for (long long primeMultiple = prime * 2; ((!m_stop_work) && (primeMultiple < m_sieveSize)); primeMultiple += prime)
                {
                    m_storage.markNumberAsNotPrime(primeMultiple);
                }
            }
        } // namespace sieve
    }     // namespace derpaul
} // namespace net
