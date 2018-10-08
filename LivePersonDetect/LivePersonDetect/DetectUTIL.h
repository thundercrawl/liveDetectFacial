#pragma once

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <stdio.h>
#include <cmath>

#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"


using namespace std;
class DetectUTIL
{
public:
	DetectUTIL();
	~DetectUTIL();
	void DetectLandmark();
private:
	int openCamera();
	void markFaceLandmark(cv::Mat frame, cv::Rect_<int> face);
	void markEyeLandmark(cv::Mat frame, cv::Rect_<int> eye);
	void detectFaces(cv::Mat&, vector<cv::Rect_<int> >&, string);
	void detectEyes(cv::Mat&, vector<cv::Rect_<int> >&, string);
	void detectNose(cv::Mat&, vector<cv::Rect_<int> >&, string);
	void detectMouth(cv::Mat&, vector<cv::Rect_<int> >&, string);
	void detectFacialFeaures(cv::Mat&, const vector<cv::Rect_<int> >, string, string, string);

	void checkVariance(vector<cv::Rect_<int>>&, cv::Rect_<int>);
	void checkVariance(cv::Rect_<int> nextrect);
	/*process functions*/
	bool isInCheckingLiveProcess();
	void setInCheckingLiveProcess(bool);
	bool inprocess = false;
	string input_image_path;
	string face_cascade_path, eye_cascade_path, nose_cascade_path, mouth_cascade_path;
	cv::Rect_<int> firstEye;

};

