#pragma once

#include <map>
#include <stdlib.h>
#include <string>

class CDataStorage
{
public:
	CDataStorage(void);
	bool isNumberMarked(long long position);
	void numberMark(long long position);
	void numberUnmark(long long position);
	long long findNextUnmarked(long long position);
	void clear(void);
	void dataLoad(std::string filename);
	void dataSave(std::string filename);
	void showUnmarked(long long start);
private:
	static const long long _bitsize;
	std::map<long long, long long> _storage;

	lldiv_t getStoragePosition(long long position);
};
