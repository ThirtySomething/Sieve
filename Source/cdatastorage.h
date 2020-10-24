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

#ifndef CDATASTORAGE_H
#define CDATASTORAGE_H

#include <map>
#include <stdlib.h>
#include <string>
#include <vector>

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
            /// <summary>
            /// Data container for holding the prime data
            /// </summary>
            class CDataStorage
            {
            public:
                /// <summary>
                /// Constructor
                /// </summary>
                /// <param name="sieveSize">Size of sieve</param>
                explicit CDataStorage(long long sieveSize);

                /// <summary>
                /// Clear internal memory
                /// </summary>
                void clear(void);

                /// <summary>
                /// Load prime data from file
                /// </summary>
                /// <param name="filename">Filename containing prime data</param>
                /// <returns>Prime where to restart</returns>
                std::tuple<long long, long long> dataLoad(std::string filename);

                /// <summary>
                /// Save prime data to file
                /// </summary>
                /// <param name="filename">Filename to save data to</param>
                /// <param name="latestPrime">Latest determined prime</param>
                /// <param name="sieveSize">Upper border of sieve</param>
                void dataSave(std::string filename, long long latestPrime, long long sieveSize);

                /// <summary>
                /// Export all primes to file
                /// </summary>
                /// <param name="filename">File to save to</param>
                /// <param name="latestPrime">Latest prime to save</param>
                void exportPrimes(std::string filename, long long latestPrime);

                /// <summary>
                /// Search for next prime
                /// </summary>
                /// <param name="number">Number to number for search of next prime</param>
                /// <returns>Next prime</returns>
                long long findNextPrime(long long number);

                /// <summary>
                /// Returns a pointer to the internal storage vector
                /// </summary>
                /// <returns>Pointer to the internal storage vector</returns>
                char* getStoragePointer(void);

                /// <summary>
                /// Returns the internal size of the storage
                /// </summary>
                /// <returns>Internal storage size</returns>
                long long getStorageSize(void);

                /// <summary>
                /// Check if given number is NOT a prime
                /// </summary>
                /// <param name="number">Number to check as prime</param>
                /// <returns>true if prime, otherwise false</returns>
                bool isNumberNotPrime(long long number);

                /// <summary>
                /// Mark number as NOT prime
                /// </summary>
                /// <param name="number">Number to mark</param>
                void markNumberAsNotPrime(long long number);

                /// <summary>
                /// Internal static variable for 'marked as prime'
                /// </summary>
                static const char m_unset;

                /// <summary>
                /// Internal static variable for not 'marked as prime'
                /// </summary>
                static const char m_set;

                /// <summary>
                /// Internal storage for primes
                /// </summary>
                std::vector<char> m_storage;

            private:
                /// <summary>
                /// Storagesize - may be different than the sieve size
                /// </summary>
                long long m_storageSize;
            };
        } // namespace sieve
    }     // namespace derpaul
} // namespace net

#endif // CDATASTORAGE_H
