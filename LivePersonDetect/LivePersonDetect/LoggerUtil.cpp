#include "LoggerUtil.h"

using namespace std;
 LoggerUtil* LoggerUtil::_instance = 0;
 std::ofstream* LoggerUtil::flog = 0;
 std::ofstream* LoggerUtil::ferrlog = 0;


LoggerUtil::~LoggerUtil()
{
	flog->close();
	ferrlog->close();
	delete flog;
	delete ferrlog;
	
}


LoggerUtil* LoggerUtil::getInstance()
{
	if (_instance == NULL)
	{
		flog = new ofstream();
		ferrlog = new ofstream();
		
		
		_instance = new LoggerUtil();
	}
	return _instance;
}
void LoggerUtil::logInfo(const char* format...)
{
	
	char buffer[1024]; 
	
	va_list args;
	va_start(args, format);
	vsprintf(buffer, format, args);
	
	va_end(args);
	
	*flog << "[ " << std::this_thread::get_id() << " ] " << getDateTime() << " "<< buffer << endl;
}
void LoggerUtil::logger(std::string out)
{
	*flog << "[ " << std::this_thread::get_id() << " ] " << getDateTime() <<" " <<out << endl;
}
LoggerUtil::LoggerUtil()
{
	flog->open("./publisher.log", fstream::out |ios::app);
	ferrlog->open("./publisher.log", fstream::out | ios::app);
}

std::string LoggerUtil::getDateTime()
{
	time_t rawtime;
	struct tm * timeinfo;
	char buffer[80];

	time(&rawtime);
	timeinfo = localtime(&rawtime);

	strftime(buffer, sizeof(buffer), "%d-%m-%Y %H:%M:%S", timeinfo);
	std::string str(buffer);

	return str;
}
