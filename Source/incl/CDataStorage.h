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

#pragma once

#include <map>
#include <stdlib.h>
#include <string>

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
                CDataStorage(void);

                /// <summary>
                /// Check if given number is NOT a prime
                /// </summary>
                /// <param name="number">Number to check as prime</param>
                /// <returns>true if prime, otherwise false</returns>
                bool isNumberPrime(long long number);

                /// <summary>
                /// Mark number as NOT prime
                /// </summary>
                /// <param name="number">Number to mark</param>
                void markNumberAsNotPrime(long long number);

                /// <summary>
                /// Search for next prime
                /// </summary>
                /// <param name="number">Number to number for search of next prime</param>
                /// <returns>Next prime</returns>
                long long findNextPrime(long long number);

                /// <summary>
                /// Clear internal memory
                /// </summary>
                void clear(void);

                /// <summary>
                /// Load prime data from file
                /// </summary>
                /// <param name="filename">Filename containing prime data</param>
                void dataLoad(std::string filename);

                /// <summary>
                /// Save prime data to file
                /// </summary>
                /// <param name="filename">Filename to save data to</param>
                void dataSave(std::string filename);

                /// <summary>
                /// Show all found primes up to given border
                /// </summary>
                /// <param name="number">Upper border</param>
                void showPrimes(long long number);

            private:
                /// <summary>
                /// Internal static variable to memorize numbers of bits
                /// </summary>
                static const long long _bitsize;

                /// <summary>
                /// Internal storage for primes
                /// </summary>
                std::map<long long, long long> _storage;

                /// <summary>
                /// Internal method to map large number to index and position
                /// </summary>
                /// <param name="number">Number to map</param>
                /// <returns>Structure with quote and remainder</returns>
                lldiv_t getStoragePosition(long long number);
            };
        }
    }
}
