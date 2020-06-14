#include "CSieve.h"
#include <chrono>
#include <iostream>

#define FILENAME "primes.txt"
#define LIMIT 1000000

int main()
{
    CSieve mySieve;

    auto start = std::chrono::steady_clock::now();
    mySieve.action(LIMIT);
    auto end = std::chrono::steady_clock::now();
    std::cout << "Time for calculation: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    //mySieve.showPrimes(LIMIT);
    //mySieve.dataSave(FILENAME);
    //mySieve.dataLoad(FILENAME);
    //mySieve.showPrimes(LIMIT);
}
