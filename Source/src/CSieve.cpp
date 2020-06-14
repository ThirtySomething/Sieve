#include "CSieve.h"
#include <iostream>

CSieve::CSieve(void)
{
}

void CSieve::action(long long maxsize)
{
	_storage.clear();
	_storage.numberMark(0);
	_storage.numberMark(1);
	long long currentprime = 2L;
	while (currentprime < maxsize)
	{
		// std::cout << "Current prime: " << currentprime << std::endl;
		markOthers(currentprime, maxsize);
		currentprime = _storage.findNextUnmarked(currentprime);
	}
}

void CSieve::dataLoad(std::string filename)
{
	_storage.dataLoad(filename);
}

void CSieve::dataSave(std::string filename)
{
	_storage.dataSave(filename);
}

void CSieve::showPrimes(long long maxsize)
{
	_storage.showUnmarked(maxsize);
}

void CSieve::markOthers(long long number, long long maxsize)
{
	for (long long current = number * 2; current < maxsize; current += number)
	{
		_storage.numberMark(current);
	}
}