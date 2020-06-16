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

#include "CDataStorage.h"
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
            /// Contains logic of sieve
            /// </summary>
            class CSieve
            {
            public:
                /// <summary>
                /// Constructor
                /// </summary>
                CSieve(void);

                /// <summary>
                /// Performs the sieve algorithm
                /// </summary>
                /// <param name="maxsize">Maximum prime to sieve to</param>
                void action(long long maxsize);

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

            private:
                /// <summary>
                /// Internal data storage of sieve
                /// </summary>
                CDataStorage _storage;

                /// <summary>
                /// Mark multiple value of given prime up to max size of sieve
                /// </summary>
                /// <param name="prime">Prime to mark multiple values</param>
                /// <param name="maxsize">Maximum number of sieve</param>
                void markMultiplePrimes(long long prime, long long maxsize);
            };
        }
    }
}
