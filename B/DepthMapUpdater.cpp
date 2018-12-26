#include "DepthMapUpdater.h"

int DepthMapUpdater::init(cv::Mat& masterBackground, cv::Mat& slaveBackground, cv::Mat& depthBackground)
{
	_backMaster = masterBackground;
	_backSlave = slaveBackground;
	_backDepth = depthBackground;
	_gpu_backMaster.upload(_backMaster);
	_gpu_backSlave.upload(_backSlave);
	_gpu_backDepth.upload(_backDepth);
	_frameCount = 0;

	_mog = cv::cuda::createBackgroundSubtractorMOG(70);
	_gauss = cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size(5, 5), 0);
#if (SELECTCHANNEL==1 | SELECTCHANNEL==2) // use Intensity or Gradient image    
	_dep = depthmap(CV_32FC1, 1, cv::IMREAD_GRAYSCALE);
#elif (SELECTCHANNEL==3) // use RGB image
	_dep = depthmap(CV_32FC3, 3, cv::IMREAD_COLOR);
#endif

	_dep.init_depth(_backMaster, _backSlave);
	_mog->apply(_gpu_backMaster, _gpu_mask, 0.01);
    return 0;
}

int DepthMapUpdater::update(cv::Mat& masterMat, cv::Mat& slaveMat, cv::Mat& depthWithMask)
{
    return 0;
}

int DepthMapUpdater::getFrameCount()
{
	return _frameCount;
}