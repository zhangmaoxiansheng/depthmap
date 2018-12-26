#include "DepthMapUpdater.h"

int DepthMapUpdater::init(cv::Mat& masterBackground, cv::Mat& slaveBackground, cv::Mat& depthBackground)
{
	_backMaster = masterBackground;
	_backSlave = slaveBackground;
	_backDepth = depthBackground;
	cv::resize(_backMaster, _backMaster, cv::Size(JIANING_WIDTH, JIANING_HEIGHT));
	cv::resize(_backSlave, _backSlave, cv::Size(JIANING_WIDTH, JIANING_HEIGHT));
	cv::resize(_backDepth, _backDepth, cv::Size(JIANING_WIDTH, JIANING_HEIGHT));
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
	cv::Mat _m, _s;
	cv::cuda::GpuMat _gpu_m;
	cv::resize(masterMat, _m, cv::Size(JIANING_WIDTH, JIANING_HEIGHT));
	cv::resize(slaveMat, _s, cv::Size(JIANING_WIDTH, JIANING_HEIGHT));
	_gpu_m.upload(_m);
	_mog->apply(_gpu_m, _gpu_mask, 0.01);
	_gauss->apply(_gpu_mask, _gpu_mask);
	_gpu_mask.download(_mask);

	//std::vector<cv::Rect> result = _elem.Find_location(_mask);
	_depth = _dep.get_depth(_m, _s);
	cv::Mat diff_mask = _elem.refine_mask(_backMaster, _m, _mask);
	depthWithMask = _dep.update_depth_robust(_depth, diff_mask);
	_frameCount++;
    return 0;
}

int DepthMapUpdater::getFrameCount()
{
	return _frameCount;
}