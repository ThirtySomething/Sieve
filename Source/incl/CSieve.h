#pragma once

#include "CDataStorage.h"
#include <string>

class CSieve
{
public:
	CSieve(void);
	void action(long long maxsize);
	void dataLoad(std::string filename);
	void dataSave(std::string filename);
	void showPrimes(long long maxsize);
private:
	CDataStorage _storage;
	void markOthers(long long number, long long maxsize);
};

