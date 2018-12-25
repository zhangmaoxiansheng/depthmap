/**
@brief DepthMapUpdater.h 
class for depth map update
@author zhu-ty
@date Dec 25, 2018
*/

#ifndef __DEPTH_MAP_UPDATER__
#define __DEPTH_MAP_UPDATER__

#include <opencv2/core/utility.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudalegacy.hpp>
#include <opencv2/video.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <sys/time.h>
#include <fstream>
#include "depthmap.h"    
#include "oflow.h"
#include "GhostElemer.h"

#define SELECTMODE 2
#define SELECTCHANNEL 1

class DepthMapUpdater
{
public:
    DepthMapUpdater(){};
    ~DepthMapUpdater(){};
    /**
	@brief init updater with background infomation
	@param cv::Mat& masterBackground: A Pano Render Camera
	@param cv::Mat& slaveBackground: texture width
	@param cv::Mat& depthBackground: texture height
	@return int(0)
	*/
    int init(cv::Mat& masterBackground, cv::Mat& slaveBackground, cv::Mat& depthBackground);
    int update(cv::Mat& masterMat, cv::Mat& slaveMat, cv::Mat& depthWithMask);
    int getFrameCount();
private:
    int frameCount = 0;
    cv::Ptr<cv::BackgroundSubtractor> mog;
    cv::Ptr<cv::cuda::Filter> gauss
    GhostElemer elem;

#if (SELECTCHANNEL==1 | SELECTCHANNEL==2) // use Intensity or Gradient image    
    depthmap dep(CV_32FC1,1,IMREAD_GRAYSCALE);
#elif (SELECTCHANNEL==3) // use RGB image
    depthmap dep(CV_32FC3,3,IMREAD_COLOR);    
#endif

    
};

#endif //__DEPTH_MAP_UPDATER__