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
			// *****************************************************************************
			CSieve::CSieve(long long maxsize) :
				m_maxSize(maxsize),
                m_currentPrime(1LL),
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
            void CSieve::sievePrimes(std::function<void(long long)> updatePrime)
			{
                m_stop_work = false;

				while (!m_stop_work && (m_currentPrime < m_maxSize))
				{
                    m_currentPrime = m_storage.findNextPrime(m_currentPrime);
                    updatePrime(m_currentPrime);
                    markPrimeMultiples(m_currentPrime);
				}
			}

			// *****************************************************************************
			// *****************************************************************************
			void CSieve::dataLoad(std::string filename)
			{
                auto[currentPrime, maxSize] = m_storage.dataLoad(filename);
                m_currentPrime = currentPrime;
                m_maxSize = maxSize;
			}

			// *****************************************************************************
			// *****************************************************************************
			void CSieve::dataSave(std::string filename)
			{
                m_storage.dataSave(filename, m_currentPrime, m_maxSize);
			}

			// *****************************************************************************
			// *****************************************************************************
			void CSieve::showPrimes(long long maxsize)
			{
				m_storage.showPrimes(maxsize);
			}

			// *****************************************************************************
			// *****************************************************************************
            void CSieve::markPrimeMultiples(long long prime)
			{
				for (long long current = prime * 2; current < m_maxSize; current += prime)
				{
					m_storage.markNumberAsNotPrime(current);
				}
			}

            // *****************************************************************************
            // *****************************************************************************
            void CSieve::interruptSieving(void)
            {
                m_stop_work = true;
            }

            // *****************************************************************************
            // *****************************************************************************
            void CSieve::initStorage(void)
            {
                m_storage.clear();
                m_storage.markNumberAsNotPrime(0LL);
                m_storage.markNumberAsNotPrime(1LL);
            }
		}
	}
}
