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

#include "cdatastorage.h"
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
            const long long CDataStorage::m_bitsize = 64L;

            // *****************************************************************************
            // *****************************************************************************
            CDataStorage::CDataStorage(void)
            {
                clear();
            }

            // *****************************************************************************
            // *****************************************************************************
            bool CDataStorage::isNumberNotPrime(long long number)
            {
                lldiv_t internalPosition = getStoragePosition(number);
                long long element = m_storage[internalPosition.quot];
                bool result = (element >> internalPosition.rem) & 1LL;
                return result;
            }

            // *****************************************************************************
            // *****************************************************************************
            void CDataStorage::markNumberAsNotPrime(long long number)
            {
                lldiv_t internalPosition = getStoragePosition(number);
                long long element = m_storage[internalPosition.quot];
                element |= 1LL << internalPosition.rem;
                m_storage[internalPosition.quot] = element;
            }

            // *****************************************************************************
            // *****************************************************************************
            long long CDataStorage::findNextPrime(long long number)
            {
                long long startValue = number + 1;
                while (isNumberNotPrime(startValue))
                {
                    startValue++;
                }
                return startValue;
            }

            // *****************************************************************************
            // *****************************************************************************
            void CDataStorage::clear(void)
            {
                m_storage.clear();
            }

            // *****************************************************************************
            // *****************************************************************************
            std::tuple<long long, long long> CDataStorage::dataLoad(std::string filename)
            {
                std::ifstream infile(filename);
                m_storage.clear();
                long long index, bits, currentPrime, maxSize;
                infile >> currentPrime;
                infile >> maxSize;
                while (infile >> index >> bits)
                {
                    m_storage[index] = bits;
                }
                infile.close();

                return {currentPrime, maxSize};
            }

            // *****************************************************************************
            // *****************************************************************************
            void CDataStorage::dataSave(std::string filename, long long currentPrime, long long maxSize)
            {
                std::ofstream myfile;
                myfile.open(filename);

                myfile << currentPrime << std::endl;
                myfile << maxSize << std::endl;
                for (auto [idx, storagePart] : m_storage)
                {
                    myfile << idx << " " << storagePart << std::endl;
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
                    if (!isNumberNotPrime(current))
                    {
                        std::cout << "Prime: " << current << std::endl;
                    }
                }
            }

            // *****************************************************************************
            // *****************************************************************************
            lldiv_t CDataStorage::getStoragePosition(long long number)
            {
                lldiv_t result = lldiv(number, CDataStorage::m_bitsize);
                return result;
            }
        }
    }
}
