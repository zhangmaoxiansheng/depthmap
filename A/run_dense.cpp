#include "opencv2/core/utility.hpp"
#include "opencv2/cudabgsegm.hpp"
#include "opencv2/cudalegacy.hpp"
#include "opencv2/video.hpp"
#include "opencv2/cudawarping.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudafilters.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <sys/time.h>
#include <fstream>
#include "depthmap.h"    
#include "oflow.h"

using namespace std;
using namespace cv;
using namespace cv::cuda;
//input : ./run_DE_INT test2.avi test2b.avi
//output: bgdepth bgimg 
int main(int argc, char** argv)
{
	char *file = argv[1];//video input 1
	char *file2 = argv[2];//video input 2
	bool useCamera = false;
	VideoCapture cap;
	VideoCapture cap2;
	if (useCamera)
		cap.open(0);
	else
		{
			cap.open(file);
			cap2.open(file2);
		}
	if (!cap.isOpened() || !cap2.isOpened())
	{
		cerr << "can not open camera or video file" << endl;
		return -1;
	}
	Mat frame;
	Mat frame2;
	cap >> frame;
	cap2 >> frame2;
	GpuMat d_frame(frame);
	GpuMat d_frame2(frame2);
	Ptr<BackgroundSubtractor> mog = cuda::createBackgroundSubtractorMOG(300);
	Ptr<BackgroundSubtractor> mog2 = cuda::createBackgroundSubtractorMOG(300);
	GpuMat d_fgmask;
	GpuMat d_fgmask2;
	GpuMat d_bgimg;
	GpuMat d_bgimg2;
	Mat bgimg;
	Mat bgimg2;
    Mat bg_depth;
    //------------------------------------------------------------parpare for DIS-----------------------------------------------------------
    int rpyrtype, nochannels, incoltype;
    incoltype = IMREAD_COLOR;
    rpyrtype = CV_32FC3;
    nochannels = 3;      
  // *** Parse rest of parameters, See oflow.h for definitions.
    if(frame.empty()||frame2.empty())
    {
        cout<<"the first frame is empty!"<<endl;
        return -1;
    }
    depthmap dep(rpyrtype,nochannels,incoltype);
	// MOG:
	mog->apply(d_frame, d_fgmask, 0.01);
	mog2->apply(d_frame2,d_fgmask2,0.01);
	for (int i = 0;i < 320;i++)
	{
		cap >> frame;
		cap2 >> frame2;
		if (frame.empty())
			break;
		cv::resize(frame, frame, Size(1000, 750));
		cv::resize(frame2,frame2,Size(1000,750));
		d_frame.upload(frame);
		d_frame2.upload(frame2);
		int64 start = cv::getTickCount();
		//update the model
		mog->apply(d_frame, d_fgmask, 0.01);
		mog->getBackgroundImage(d_bgimg);
		mog2->apply(d_frame2, d_fgmask2, 0.01);
		mog2->getBackgroundImage(d_bgimg2);
		d_bgimg.download(bgimg);
		d_bgimg2.download(bgimg2);
		double fps = cv::getTickFrequency() / (cv::getTickCount() - start);
		std::cout << "time : " << 1000/fps << std::endl;
		int key = waitKey(10);
		if (key == 27)
			break;
		
	}
	bg_depth = dep.init_depth(bgimg,bgimg2);
	imshow("bg1",bgimg);
	imshow("bg2",bgimg2);
	cv::waitKey(0);
	//output bg_depth bgimg
	return 0;
}

