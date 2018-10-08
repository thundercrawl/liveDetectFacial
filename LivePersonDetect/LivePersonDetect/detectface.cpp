/**
@file videocapture_basic.cpp
@brief A very basic sample for using VideoCapture and VideoWriter
@author PkLab.net
@date Aug 24, 2016
*/

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>

#include <dlib/image_processing.h>
#include <opencv2/imgproc.hpp>  // Gaussian Blur
#include <opencv2/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>  // OpenCV window I/O
#include <opencv2/features2d.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <dlib/opencv/cv_image.h>

//#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
using namespace dlib;
using namespace cv;
using namespace std;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET> using ares = relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128, avg_pool_everything<
	alevel0<
	alevel1<
	alevel2<
	alevel3<
	alevel4<
	max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2,
	input_rgb_image_sized<150>
	>>>>>>>>>>>>;

std::vector<matrix<rgb_pixel>> jitter_image(
	const matrix<rgb_pixel>& img
);

class CascadeDetectorAdapter : public DetectionBasedTracker::IDetector
{
public:
	CascadeDetectorAdapter(cv::Ptr<cv::CascadeClassifier> detector) :
		IDetector(),
		Detector(detector)
	{
		CV_Assert(detector);
	}

	void detect(const cv::Mat &Image, std::vector<cv::Rect> &objects)
	{
		Detector->detectMultiScale(Image, objects, scaleFactor, minNeighbours, 0, minObjSize, maxObjSize);
	}

	virtual ~CascadeDetectorAdapter()
	{}

private:
	CascadeDetectorAdapter();
	cv::Ptr<cv::CascadeClassifier> Detector;
};

static cv::Rect dlibRectangleToOpenCV(dlib::rectangle r)
{
	return cv::Rect(cv::Point2i(r.left(), r.top()), cv::Point2i(r.right() + 1, r.bottom() + 1));
}

static dlib::rectangle openCVRectToDlib(cv::Rect r)
{
	return dlib::rectangle((long)r.tl().x, (long)r.tl().y, (long)r.br().x - 1, (long)r.br().y - 1);
}

static void checkResNet()
{

	anet_type net;
	deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;
}
int mainx();
int maind(void)
{

		cv::VideoCapture vcap;
		cv::Mat image;

		// This works on a D-Link CDS-932L
		const std::string videoStreamAddress = "http://<admin:xgPassw0rd>@<192.168.1.64>/video.cgi?.mjpg";

		//open the video stream and make sure it's opened
		if (!vcap.open(videoStreamAddress)) {
			std::cout << "Error opening video stream or file" << std::endl;
			return -1;
		}

		for (;;) {
			if (!vcap.read(image)) {
				std::cout << "No frame" << std::endl;
				cv::waitKey();
			}
			cv::imshow("Output Window", image);
			if (cv::waitKey(1) >= 0) break;
		}
	
	
	return 1;
}
int mainx()
{
	namedWindow("Detect face");
	
	shape_predictor sp;
	cout << "load landmark" << endl;
	deserialize("./shape_predictor_68_face_landmarks.dat") >> sp;
	cout << "load video" << endl;
	anet_type net;
	deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;
	cout << "load res net" << endl;
	VideoCapture VideoStream(0);
	
	if (!VideoStream.isOpened())
	{
		printf("Error: Cannot open video stream from camera\n");
		return 1;
	}

	std::string cascadeFrontalfilename = "./data/lbpcascades/lbpcascade_frontalface.xml";
	cv::Ptr<cv::CascadeClassifier> cascade = makePtr<cv::CascadeClassifier>(cascadeFrontalfilename);
	cv::Ptr<DetectionBasedTracker::IDetector> MainDetector = makePtr<CascadeDetectorAdapter>(cascade);
	if (cascade->empty())
	{
		printf("Error: Cannot load %s\n", cascadeFrontalfilename.c_str());
		return 2;
	}
	cout << "load cascade";

	cascade = makePtr<cv::CascadeClassifier>(cascadeFrontalfilename);
	cv::Ptr<DetectionBasedTracker::IDetector> TrackingDetector = makePtr<CascadeDetectorAdapter>(cascade);
	if (cascade->empty())
	{
		printf("Error: Cannot load %s\n", cascadeFrontalfilename.c_str());
		return 2;
	}

	DetectionBasedTracker::Parameters params;
	DetectionBasedTracker Detector(MainDetector, TrackingDetector, params);

	if (!Detector.run())
	{
		printf("Error: Detector initialization failed\n");
		return 2;
	}
	cout << "load detector";

	Mat ReferenceFrame;
	Mat GrayFrame;
	std::vector<Rect> Faces;
	bool saveImge = false;
	dlib::int64 t0 = cv::getTickCount();
	Mat img1 = cv::imread("./images/4.jpg",1);
	//cv::imshow("detect face1", img1);
	Mat img1gray;
	cvtColor(img1, img1gray, CV_BGR2GRAY);
	//cv::imshow("detect face1", img1gray);
	
	//matrix<rgb_pixel> img1;
	//load_image(img1, "./images/1.jpg");
	array2d<unsigned char> imgdlib1;
	load_image(imgdlib1, "./images/4.jpg");
	//pyramid_up(imgdlib1);
	frontal_face_detector dlibdetector = get_frontal_face_detector();
	std::vector<dlib::rectangle> dets = dlibdetector(imgdlib1);

	if (dets.size() == 0)
	{
		cout << "dlib detector not find any faces" << endl;
	}
	else
	{
		cout << "dlib find the faces in img,size="<<dets.size() << endl;
		matrix<rgb_pixel> face_chip1;
		auto shape1 = sp(imgdlib1, dets[0]);
		dlib::extract_image_chip(imgdlib1, get_face_chip_details(shape1, 150, 0.25), face_chip1);
		cv::Mat chipMat = dlib::toMat(face_chip1);
		cv::imwrite("commpare.jpg", chipMat);
		std::vector<matrix<rgb_pixel>> faces;
		std::vector<matrix<float, 0, 1>> face_descriptors = net(faces);
		do
		{
			VideoStream >> ReferenceFrame;
			Mat converImg;
			converImg = ReferenceFrame;
			//CV_BGR2GRAY ,COLOR_RGB2GRAY
			cvtColor(ReferenceFrame, GrayFrame, CV_BGR2GRAY);
			Detector.process(GrayFrame);
			Detector.getObjects(Faces);

			bool ready = false;;
			const char* file_name;
			for (size_t i = 0; i < Faces.size(); i++)
			{

				if (getTickCount() - t0>1000000 * 10)
				{
					ostringstream convert;
					t0 = cv::getTickCount();
					convert << "./images/image" << t0 << ".jpg";
					file_name = convert.str().c_str();
					//imwrite(file_name, ReferenceFrame);
					ready = true;
				}
				std::vector<full_object_detection> shapes;
				dlib::rectangle rect;
				rect = openCVRectToDlib(Faces[i]);
				dlib::array2d<bgr_pixel> dlibImage;
				dlib::assign_image(dlibImage, dlib::cv_image<bgr_pixel>(ReferenceFrame));
				if (ready)
				{
				}
				full_object_detection shape = sp(dlibImage, rect);
				auto shape2 = sp(dlibImage, rect);


				matrix<rgb_pixel> face_chip2;


				dlib::extract_image_chip(dlibImage, get_face_chip_details(shape2, 150, 0.25), face_chip2);
				cv::Mat chipMat = dlib::toMat(face_chip2);
				//cv::imwrite( "file.jpg", chipMat);
				cout << "number of parts: " << shape.num_parts() << endl;
				cout << "pixel position of first part:  " << shape.part(0) << endl;
				cout << "pixel position of second part: " << shape.part(67) << endl;
				// You get the idea, you can get all the face part locations if
				// you want them.  Here we just store them in shapes so we can
				// put them on the screen.
				shapes.push_back(shape);

				std::vector<matrix<float, 0, 1>> face_descriptors = net(faces);


				// In particular, one simple thing we can do is face clustering.  This next bit of code
				// creates a graph of connected faces and then uses the Chinese whispers graph clustering
				// algorithm to identify how many people there are and which faces belong to whom.
				std::vector<sample_pair> edges;
				for (size_t i = 0; i < face_descriptors.size(); ++i)
				{
					for (size_t j = i + 1; j < face_descriptors.size(); ++j)
					{
						// Faces are connected in the graph if they are close enough.  Here we check if
						// the distance between two face descriptors is less than 0.6, which is the
						// decision threshold the network was trained to use.  Although you can
						// certainly use any other threshold you find useful.
						if (length(face_descriptors[i] - face_descriptors[j]) < 0.6)
							edges.push_back(sample_pair(i, j));
					}
				}
				converImg = dlib::toMat(dlibImage);
				for (unsigned long i = 0; i < shape.num_parts(); ++i)
				{
					cv::circle(converImg, Point(shape.part(i).x(), shape.part(i).y()), 1.0, Scalar(240, 255, 255), 1, 8);
					//cv::ellipse(ReferenceFrame,Point(shape.part(i).x, shape.part(i).y), Size(2.0, 3.0), 45, 0, 360, Scalar(255, 0, 0), 1, 8);

				}

				//cv::rectangle(ReferenceFrame, Faces[i], Scalar(0, 255, 0));
			}

			imshow("detect face", converImg);
		} while (waitKey(30) < 0);

	}
	cout << "load image1" <<endl;
	//cv::Mat similar_img;
	//similar_img = dlib::toMat(img1);
	Detector.process(img1gray);
	std::vector<Rect> image1Faces;
	Detector.getObjects(image1Faces);
	if (image1Faces.size() == 0)
	{
		cout << "face not found in image1 by opencv" << endl;

	}
	else
	{
		cout << "opencv face size=" << image1Faces.size() << endl;

		matrix<rgb_pixel> face_chip1;

		dlib::rectangle rect1;
		rect1 = openCVRectToDlib(image1Faces[0]);
		dlib::array2d<bgr_pixel> dlibimg1;
		dlib::assign_image(dlibimg1, dlib::cv_image<bgr_pixel>(img1));
		auto shape1 = sp(dlibimg1, rect1);
		dlib::extract_image_chip(dlibimg1, get_face_chip_details(shape1, 150, 0.25), face_chip1);
		cv::Mat chipMat = dlib::toMat(face_chip1);
		cv::imwrite("commpare.jpg", chipMat);
	}
	do
	{
		VideoStream >> ReferenceFrame;
		Mat converImg;
		converImg = ReferenceFrame;
		//CV_BGR2GRAY ,COLOR_RGB2GRAY
		cvtColor(ReferenceFrame, GrayFrame, CV_BGR2GRAY);
		Detector.process(GrayFrame);
		Detector.getObjects(Faces);
		
		bool ready = false;;
		const char* file_name;
		for (size_t i = 0; i < Faces.size(); i++)
		{
			
			if(getTickCount()-t0>1000000*10)
			{ 
				ostringstream convert;
				t0 = cv::getTickCount();
				convert << "./images/image" << t0 << ".jpg";
				file_name = convert.str().c_str();
				//imwrite(file_name, ReferenceFrame);
				ready = true;
			}
			std::vector<full_object_detection> shapes;
			dlib::rectangle rect;
			rect = openCVRectToDlib(Faces[i]);
			dlib::array2d<bgr_pixel> dlibImage;
			dlib::assign_image(dlibImage, dlib::cv_image<bgr_pixel>(ReferenceFrame));
			if (ready)
			{	
			}
				full_object_detection shape = sp(dlibImage, rect);
				auto shape2=sp(dlibImage, rect);
				
				
				matrix<rgb_pixel> face_chip2;
				
				
				dlib::extract_image_chip(dlibImage, get_face_chip_details(shape2, 150, 0.25), face_chip2);
				cv::Mat chipMat = dlib::toMat(face_chip2);
				//cv::imwrite( "file.jpg", chipMat);
				cout << "number of parts: " << shape.num_parts() << endl;
				cout << "pixel position of first part:  " << shape.part(0) << endl;
				cout << "pixel position of second part: " << shape.part(67) << endl;
				// You get the idea, you can get all the face part locations if
				// you want them.  Here we just store them in shapes so we can
				// put them on the screen.
				shapes.push_back(shape);
				
				converImg = dlib::toMat(dlibImage);
				for (unsigned long i = 0; i < shape.num_parts(); ++i)
				{
					cv::circle(converImg, Point(shape.part(i).x(), shape.part(i).y()), 1.0, Scalar(240, 255, 255), 1, 8);
					//cv::ellipse(ReferenceFrame,Point(shape.part(i).x, shape.part(i).y), Size(2.0, 3.0), 45, 0, 360, Scalar(255, 0, 0), 1, 8);

				}
				
			//cv::rectangle(ReferenceFrame, Faces[i], Scalar(0, 255, 0));
		}

		imshow("detect face", converImg);
	} while (waitKey(30) < 0);

	Detector.stop();

	return 0;
	
	
}

int captureAndDrawRect()
{
	Mat frame;
	//--- INITIALIZE VIDEOCAPTURE
	VideoCapture cap;
	// open the default camera using default API
	cap.open(0);
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

	//--- GRAB AND WRITE LOOP
	cout << "Start grabbing" << endl
		<< "Press any key to terminate" << endl;
	for (;;)
	{
		// wait for a new frame from camera and store it into 'frame'
		cap.read(frame);
		// check if we succeeded
		if (frame.empty()) {
			cerr << "ERROR! blank frame grabbed\n";
			break;
		}
		Mat queryToShow;
		//show small frame on the results display
		resize(frame, queryToShow, Size(0, 0), 0.25, 0.25);
		cv::rectangle(frame, Point(20, 20), Point(400, 400), (0, 255, 0), 2);
		putText(frame, "jimmy", Point(30, 30), FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 1);
		// show live and wait for a key with timeout long enough to show images
		imshow("Live", frame);

		if (waitKey(10) >= 0)
			break;

	}
	cap.release();
	destroyAllWindows();
	return 0;
}