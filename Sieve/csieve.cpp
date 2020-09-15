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
				m_currentPrime(0LL),
				m_abort_thread(false),
				m_stop_work(false)
			{
				m_thread_future = std::async(std::launch::async, [&]() {
					while (!m_abort_thread)
					{
						auto keystate = ::GetAsyncKeyState(VK_ESCAPE);
						if (keystate & ((short)1 << (sizeof(short) * 8 - 1)))
						{
							std::cout << "ESC hit, wait until abort" << std::endl;
							m_stop_work = true;
							break;
						}

						std::this_thread::sleep_for(std::chrono::milliseconds(50));
					}
				});
			}

			// *****************************************************************************
			// *****************************************************************************
			CSieve::~CSieve()
			{
				m_abort_thread = true;
				m_thread_future.wait();
			}

			// *****************************************************************************
			// *****************************************************************************
			void CSieve::sievePrimes(void)
			{
				m_storage.clear();
				m_storage.markNumberAsNotPrime(0LL);
				m_storage.markNumberAsNotPrime(1LL);
				m_currentPrime = 2LL;

				while (!m_stop_work && (m_currentPrime < m_maxSize))
				{
					markMultiplePrimes(m_currentPrime);
					m_currentPrime = m_storage.findNextPrime(m_currentPrime);
				}
			}

			// *****************************************************************************
			// *****************************************************************************
			void CSieve::dataLoad(std::string filename)
			{
				m_currentPrime = m_storage.dataLoad(filename);
			}

			// *****************************************************************************
			// *****************************************************************************
			void CSieve::dataSave(std::string filename)
			{
				m_storage.dataSave(filename, m_currentPrime);
			}

			// *****************************************************************************
			// *****************************************************************************
			void CSieve::showPrimes(long long maxsize)
			{
				m_storage.showPrimes(maxsize);
			}

			// *****************************************************************************
			// *****************************************************************************
			void CSieve::markMultiplePrimes(long long prime)
			{
				for (long long current = prime * 2; current < m_maxSize; current += prime)
				{
					m_storage.markNumberAsNotPrime(current);
				}
			}
		}
	}
}
