#ifndef _LoggerUtil
#define _LoggerUtil
#pragma once
#include <chrono>
#include <thread>
#include <iostream>
#include <string>
#include <fstream>
#include <cstdarg>
using namespace std;
class LoggerUtil
{
public:
	
	virtual ~LoggerUtil();
	static LoggerUtil* getInstance();
	void logger(string);
	void logInfo(const char* ...);
private: 
	LoggerUtil();
	static LoggerUtil* _instance;
	LoggerUtil& operator=(LoggerUtil const& copy);
	static std::ofstream* flog;
	static std::ofstream* ferrlog;
	std::string getDateTime();
};

#endif

