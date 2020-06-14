#include "CDataStorage.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iterator>

const long long CDataStorage::_bitsize = 64;

CDataStorage::CDataStorage(void) 
{
	clear();
}

bool CDataStorage::isNumberMarked(long long position)
{
	lldiv_t internalPosition = getStoragePosition(position);
	long long element = _storage[internalPosition.quot];
	bool result = (element >> internalPosition.rem) & 1ULL;
	return result;
}

void CDataStorage::numberMark(long long position)
{
	lldiv_t internalPosition = getStoragePosition(position);
	long long element = _storage[internalPosition.quot];
	element |= 1ULL << internalPosition.rem;
	_storage[internalPosition.quot] = element;
}

void CDataStorage::numberUnmark(long long position)
{
	lldiv_t internalPosition = getStoragePosition(position);
	long long element = _storage[internalPosition.quot];
	element &= ~(1ULL << internalPosition.rem);
	_storage[internalPosition.quot] = element;
}

long long CDataStorage::findNextUnmarked(long long position)
{
	long long startValue = position + 1;
	while (isNumberMarked(startValue))
	{
		startValue++;
	}
	return startValue;
}

void CDataStorage::clear(void)
{
	_storage.clear();
}

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

void CDataStorage::dataSave(std::string filename)
{
	std::ofstream myfile;
	myfile.open(filename);

	for (std::pair<long long, long long> element : _storage) {
		myfile << element.first << " " << element.second << std::endl;
	}

	myfile.close();
}

void CDataStorage::showUnmarked(long long maximum)
{
	long long maxValue = maximum;

	for (long long current = 2L; current < maxValue; current++)
	{
		if (!isNumberMarked(current))
		{
			std::cout << "Current prime: " << current << std::endl;
		}
	}
}

lldiv_t CDataStorage::getStoragePosition(long long position)
{
	lldiv_t result = lldiv(position, CDataStorage::_bitsize);
	return result;
}
