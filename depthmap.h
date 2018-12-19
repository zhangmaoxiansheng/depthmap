#ifndef _DEPTHMAP_H_
#define _DEPTHMAP_H_

#include <opencv2/opencv.hpp>
#include "oflow.h"

using namespace std;
using namespace cv;
//using namespace cv::cuda;

class depthmap
{
public:
    //parameter
    int lv_f, lv_l, maxiter, miniter, patchsz, patnorm, costfct, tv_innerit, tv_solverit, verbosity;
    float mindprate, mindrrate, minimgerr, poverl, tv_alpha, tv_gamma, tv_delta, tv_sor;
    bool usefbcon, usetvref;
    char* outfile;
    int rpyrtype,nochannels,incoltype;
    cv::Mat img_ao_mat,img_bo_mat;
    cv::Mat img_ao_fmat, img_bo_fmat;
    cv::Size sz;
    int width_org;// = sz.width;   // unpadded original image size
    int height_org;// = sz.height;
    int padw, padh;
    int scfct;
    int div;
    depthmap(int rpyrtype,int nochannels,int incoltype);
    Mat get_depth(Mat input1,Mat input2);
    Mat update_depth(Mat bg_depth,vector<Rect> result,Mat frame,Mat frame2);
    Mat init_depth(Mat init1,Mat init2);
    void SavePFMFile(cv::Mat& img, const char* filename);
    void SaveFlowFile(cv::Mat& img, const char* filename);
    void update_mat(Mat input1,Mat input2);

private:
    void ConstructImgPyramide(const cv::Mat & img_ao_fmat, cv::Mat * img_ao_fmat_pyr, cv::Mat * img_ao_dx_fmat_pyr, cv::Mat * img_ao_dy_fmat_pyr, const float ** img_ao_pyr, const float ** img_ao_dx_pyr, const float ** img_ao_dy_pyr, const int lv_f, const int lv_l, const int rpyrtype, const bool getgrad, const int imgpadding, const int padw, const int padh);
    int AutoFirstScaleSelect(int imgwidth, int fratio, int patchsize);

};

#endif
