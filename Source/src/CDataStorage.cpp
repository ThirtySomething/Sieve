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

#include "CDataStorage.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iterator>

// *****************************************************************************
// Namespace of Sieve
// *****************************************************************************
namespace net
{
    // *****************************************************************************
    // Namespace of Sieve
    // *****************************************************************************
    namespace derpaul
    {
        // *****************************************************************************
        // Namespace of Sieve
        // *****************************************************************************
        namespace sieve
        {
            // *****************************************************************************
            // Constants
            // *****************************************************************************
            const long long CDataStorage::_bitsize = 64L;

            // *****************************************************************************
            // *****************************************************************************
            CDataStorage::CDataStorage(void)
            {
                clear();
            }

            // *****************************************************************************
            // *****************************************************************************
            bool CDataStorage::isNumberPrime(long long number)
            {
                lldiv_t internalPosition = getStoragePosition(number);
                long long element = _storage[internalPosition.quot];
                bool result = (element >> internalPosition.rem) & 1LL;
                return result;
            }

            // *****************************************************************************
            // *****************************************************************************
            void CDataStorage::markNumberAsNotPrime(long long number)
            {
                lldiv_t internalPosition = getStoragePosition(number);
                long long element = _storage[internalPosition.quot];
                element |= 1LL << internalPosition.rem;
                _storage[internalPosition.quot] = element;
            }

            // *****************************************************************************
            // *****************************************************************************
            long long CDataStorage::findNextPrime(long long number)
            {
                long long startValue = number + 1;
                while (isNumberPrime(startValue))
                {
                    startValue++;
                }
                return startValue;
            }

            // *****************************************************************************
            // *****************************************************************************
            void CDataStorage::clear(void)
            {
                _storage.clear();
            }

            // *****************************************************************************
            // *****************************************************************************
            void CDataStorage::dataLoad(std::string filename)
            {
                std::ifstream infile(filename);
                _storage.clear();
                long long index, bits;
                while (infile >> index >> bits)
                {
                    _storage[index] = bits;
                }
                infile.close();
            }

            // *****************************************************************************
            // *****************************************************************************
            void CDataStorage::dataSave(std::string filename)
            {
                std::ofstream myfile;
                myfile.open(filename);

                for (std::pair<long long, long long> element : _storage)
                {
                    myfile << element.first << " " << element.second << std::endl;
                }

                myfile.close();
            }

            // *****************************************************************************
            // *****************************************************************************
            void CDataStorage::showPrimes(long long number)
            {
                long long maxValue = number;

                for (long long current = 2L; current < maxValue; current++)
                {
                    if (!isNumberPrime(current))
                    {
                        std::cout << "Current prime: " << current << std::endl;
                    }
                }
            }

            // *****************************************************************************
            // *****************************************************************************
            lldiv_t CDataStorage::getStoragePosition(long long number)
            {
                lldiv_t result = lldiv(number, CDataStorage::_bitsize);
                return result;
            }
        }
    }
}
