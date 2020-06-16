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
#include <chrono>
#include <iostream>

#define FILENAME "primes.txt"
#define LIMIT 100

int main()
{
    net::derpaul::sieve::CSieve mySieve;

    auto start = std::chrono::steady_clock::now();
    mySieve.action(LIMIT);
    auto end = std::chrono::steady_clock::now();
    std::cout << "Time for calculation: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    mySieve.showPrimes(LIMIT);

    //mySieve.dataSave(FILENAME);
    //mySieve.dataLoad(FILENAME);
    //mySieve.showPrimes(LIMIT);
}
