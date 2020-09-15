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

#ifndef CSIEVE_H
#define CSIEVE_H

#include "cdatastorage.h"
#include <string>
#include <future>

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
            /// Contains logic of sieve
            /// </summary>
            class CSieve
            {
            public:
                /// <summary>
                /// Constructor
                /// </summary>
                /// <param name="maxsize">Maximum prime to sieve to</param>
                explicit CSieve(long long maxsize);

                /// <summary>
                /// Destructor
                /// </summary>
                ~CSieve(void);

                /// <summary>
                /// Performs the sieve algorithm
                /// <param name="updatePrime">Function called after new prime determined</param>
                /// </summary>
                void sievePrimes(std::function<void(long long)> updatePrime);

                /// <summary>
                /// Load sieve data from file
                /// </summary>
                /// <param name="filename">Filename of sieve data</param>
                void dataLoad(std::string filename);

                /// <summary>
                /// Write sieve data to file
                /// </summary>
                /// <param name="filename">Filename of sieve data</param>
                void dataSave(std::string filename);

                /// <summary>
                /// Show all primes up to given limit
                /// </summary>
                /// <param name="maxsize">Upper border</param>
                void showPrimes(long long maxsize);

                /// <summary>
                /// Set internal flag to interrupt sieve process.
                /// </summary>
                void interruptSieving(void);

            private:
                /// <summary>
                /// Internal data storage of sieve
                /// </summary>
                CDataStorage m_storage;

                /// <summary>
                /// Upper border of sieve
                /// </summary>
                long long m_maxSize;

                /// <summary>
                /// Memorize current prime for saving purposes
                /// </summary>
                long long m_currentPrime;

                /// <summary>
                /// Flag to abort sieve of primes
                /// </summary>
                bool m_stop_work;

                /// <summary>
                /// Mark all multiples of given prime up to max size of sieve
                /// </summary>
                /// <param name="prime">Prime to mark multiples</param>
                void markPrimeMultiples(long long prime);

                /// <summary>
                /// Prepare storage for usage in sieve
                /// </summary>
                void initStorage(void);
            };
        }
    }
}

#endif
