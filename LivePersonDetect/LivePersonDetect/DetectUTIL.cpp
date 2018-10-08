

#include "DetectUTIL.h"
#include "LoggerUtil.h"


DetectUTIL::DetectUTIL()
{
	cout << "using DetectUTIL" <<endl;
}


DetectUTIL::~DetectUTIL()
{
}

void DetectUTIL::setInCheckingLiveProcess(bool set)
{
	this->inprocess = set;
}
bool DetectUTIL::isInCheckingLiveProcess()
{
	return this->inprocess;
}
void DetectUTIL::checkVariance(vector<cv::Rect_<int> >& samples, cv::Rect_<int> orirect)
{
}
void DetectUTIL::checkVariance(cv::Rect_<int> nextrect)
{
	double comparable = abs(this->firstEye.x - firstEye.width) / abs(firstEye.y - firstEye.height);
	double result = abs(nextrect.x - nextrect.width) / abs(nextrect.y - nextrect.height);

	LoggerUtil::getInstance()->logger("Get the variance:"+ to_string(result)+" comparable:"+to_string(comparable));
}
void DetectUTIL::detectFaces(cv::Mat&, vector<cv::Rect_<int> >&, string)
{

}
void DetectUTIL::DetectLandmark()
{
	LoggerUtil::getInstance()->logger("Detect by landmark from Division team" );
	LoggerUtil::getInstance()->logger ("open camera");
	openCamera();

}

void DetectUTIL::detectEyes(cv::Mat& img, vector<cv::Rect_<int> >& eyes, string cascade_path)
{
	cv::CascadeClassifier eyes_cascade;
	eyes_cascade.load(cascade_path);

	eyes_cascade.detectMultiScale(img, eyes, 1.20, 5, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));

	cv::Rect_<int> firstRect;
	if (eyes.size() == 2)
	{
		if (!this->isInCheckingLiveProcess())
		{
			this->firstEye = eyes.at(0);
			this->setInCheckingLiveProcess(true);
			return;
			
		}
		for(cv::Rect_<int> rect:eyes)
			markEyeLandmark(img, rect);
	}
}
void DetectUTIL::markFaceLandmark(cv::Mat frame, cv::Rect_<int> rect)
{
	cv::rectangle(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height),
		cv::Scalar(255, 0, 0), 1, 4);

}

void DetectUTIL::markEyeLandmark(cv::Mat frame, cv::Rect_<int> rect)
{
	cv::rectangle(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height),
		cv::Scalar(255, 0, 0), 1, 4);
	this->checkVariance(rect);

}
int DetectUTIL::openCamera()
{
	cv::Mat frame;
	//--- INITIALIZE VIDEOCAPTURE
	cv::VideoCapture cap;
	// open the default camera using default API
	// cap.open(0);
	// OR advance usage: select any API backend
	int deviceID = 0;             // 0 = open default camera
	int apiID = cv::CAP_ANY;      // 0 = autodetect default API
	// open selected camera using selected API
	cap.open(deviceID + apiID);
	// check if we succeeded
	if (!cap.isOpened()) {
		cerr << "ERROR! Unable to open camera\n";
		return -1;
	}
	//use frame to detect face
	vector<cv::Rect_<int> > faces;
	detectFaces(frame, faces, face_cascade_path);
	cv::CascadeClassifier face_cascade;
	face_cascade.load("./haarcascades/haarcascade_frontalcatface.xml");
	
	for (;;)
	{
		// wait for a new frame from camera and store it into 'frame'
		cap.read(frame);
		// check if we succeeded
		if (frame.empty()) {
			cerr << "ERROR! blank frame grabbed\n";
			break;
		}
		

		face_cascade.detectMultiScale(frame, faces, 1.15, 3, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));
		if (faces.size() > 0)
		{ 
			//LoggerUtil::getInstance()->logger("Detect face"+ std::to_string(faces.begin()->height)+":"+std::to_string(faces.begin()->width));
			cv::Rect_<int> firstface = faces.at(0);
			markFaceLandmark(frame, firstface);
			//detect eyes
			vector<cv::Rect_<int> > eyes;
			detectEyes(frame, eyes, "./haarcascades/haarcascade_eye.xml");

		}

		

		// show live and wait for a key with timeout long enough to show images
		imshow("Live", frame);
		if (cv::waitKey(5) >= 0)
			break;
	}

}
