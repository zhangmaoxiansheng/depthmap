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
#include "GhostElemer.h"

using namespace std;
using namespace cv;
using namespace cv::cuda;
//input : ./run_DE_INT test2.avi test2b.avi
//output:   vector<Rect> result, Mat depth_map
int main(int argc, char** argv)
{
	char *file = argv[1];//video input 1
	char *file2 = argv[2];//video input 2
	vector<Rect> result;
	int count = 0;
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
	Mat frame_o;
    Mat init1;
    Mat init2;
	cap >> frame;
	cap2 >> frame2;


	GpuMat d_frame(frame);

	Ptr<BackgroundSubtractor> mog = cuda::createBackgroundSubtractorMOG(70);
	GpuMat d_fgmask;
	GpuMat d_fgimg;
	GpuMat d_bgimg;
	GpuMat d_ghost;//
	GpuMat d_ghost_curr;
	Ptr<cv::cuda::Filter> gauss = cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1, Size(5,5), 0);

	Mat input1;
    Mat input2;
    Mat fgmask;
	Mat fgimg;
	Mat bgimg;
    Mat bg_depth;
    Mat depth_map;
    Mat bg_old;
	GhostElemer elem;
    //------------------------------------------------------------parpare for DIS-----------------------------------------------------------

    int rpyrtype, nochannels, incoltype;
    //different version
    #if (SELECTCHANNEL==1 | SELECTCHANNEL==2) // use Intensity or Gradient image      
    incoltype = CV_LOAD_IMAGE_GRAYSCALE;        
    rpyrtype = CV_32FC1;
    nochannels = 1;
    #elif (SELECTCHANNEL==3) // use RGB image
    incoltype = CV_LOAD_IMAGE_COLOR;
    rpyrtype = CV_32FC3;
    nochannels = 3;      
    #endif
  // *** Parse rest of parameters, See oflow.h for definitions.
    if(frame.empty()||frame2.empty())
    {
        cout<<"the first frame is empty!"<<endl;
        return -1;
    }

    
    depthmap dep(rpyrtype,nochannels,incoltype);
    bg_depth = dep.init_depth(frame,frame2);
    init1 = frame.clone();
    init2 = frame2.clone();

	// MOG:
	mog->apply(d_frame, d_fgmask, 0.01);

	for (;;)
	{
		cap >> frame;
		frame_o = frame.clone();
		cap2 >> frame2;
		if (frame.empty())
			break;
		cv::resize(frame, frame, Size(1000, 750));
		d_frame.upload(frame);
		int64 start = cv::getTickCount();

		//update the model
		
		mog->apply(d_frame, d_fgmask, 0.01);
		mog->getBackgroundImage(d_bgimg);

		d_fgimg.create(d_frame.size(), d_frame.type());
		d_fgimg.setTo(Scalar::all(0));
		d_frame.copyTo(d_fgimg, d_fgmask);
		gauss->apply(d_fgmask, d_fgmask);

		if (count == 5)
			d_fgmask.copyTo(d_ghost);
		if (count > 75)
		{
			cv::cuda::subtract(d_fgmask, d_ghost, d_fgmask);
			if (!d_ghost_curr.empty())
				cv::cuda::subtract(d_fgmask, d_ghost_curr, d_fgmask);
			if (count % 5 == 0)//
				d_fgmask.copyTo(d_ghost_curr);
		}
		d_fgmask.download(fgmask);
		d_fgimg.download(fgimg);
		if (!d_bgimg.empty())
			d_bgimg.download(bgimg);
		if (count == 1)
            bg_old = bgimg;
		result = elem.Find_location(fgmask);

		if (count >= 75)
			elem.ghost_elem_update(result);
		
		cv::resize(frame_o,frame_o,Size(1000,750));
		cv::resize(frame2,frame2,Size(1000,750));
		//depth_map = dep.get_depth(frame_o,frame2);
    	depth_map = dep.update_depth(bg_depth,result,frame_o,frame2);
		double fps = cv::getTickFrequency() / (cv::getTickCount() - start);
		std::cout << "time : " << 1000/fps << std::endl;
		cout<<"depth update done!"<<endl;
		// if (count == 80)
		// 	dep.SavePFMFile(depth_map,"depth_map.pfm");
//------------------------------------------repair the background
    	if(count == 75)
    	{
			Mat bg_new;
			cv::absdiff(bgimg,bg_old,bg_new);
			cv::threshold(bg_new,bg_new,30,250,CV_THRESH_BINARY);
			cv::cvtColor(bg_new,bg_new,CV_BGR2GRAY);
			vector<Rect> ground = elem.Find_location(bg_new);
			bg_depth = dep.update_depth(bg_depth,ground,init1,init2);
    	}
		result.clear();
		count = count + 1;
		int key = waitKey(10);
		if (key == 27)
			break;
	}
	return 0;
}

