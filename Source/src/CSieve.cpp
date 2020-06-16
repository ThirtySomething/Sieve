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

#include "CSieve.h"
#include <iostream>
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
			// *****************************************************************************
			CSieve::CSieve(long long maxsize) : _maxSize(maxsize)
			{
			}

			// *****************************************************************************
			// *****************************************************************************
			void CSieve::sievePrimes(void)
			{
				_storage.clear();
				_storage.markNumberAsNotPrime(0);
				_storage.markNumberAsNotPrime(1);
				long long currentprime = 2L;
				bool abort = false;
				while (!abort && (currentprime < _maxSize))
				{
					markMultiplePrimes(currentprime);
					currentprime = _storage.findNextPrime(currentprime);
					if (GetAsyncKeyState(VK_ESCAPE) & 0x8000) 
					{
						abort = true;
					}
				}
			}

			// *****************************************************************************
			// *****************************************************************************
			void CSieve::dataLoad(std::string filename)
			{
				_storage.dataLoad(filename);
			}

			// *****************************************************************************
			// *****************************************************************************
			void CSieve::dataSave(std::string filename)
			{
				_storage.dataSave(filename);
			}

			// *****************************************************************************
			// *****************************************************************************
			void CSieve::showPrimes(long long maxsize)
			{
				_storage.showPrimes(maxsize);
			}

			// *****************************************************************************
			// *****************************************************************************
			void CSieve::markMultiplePrimes(long long prime)
			{
				for (long long current = prime * 2; current < _maxSize; current += prime)
				{
					_storage.markNumberAsNotPrime(current);
				}
			}
		}
	}
}
