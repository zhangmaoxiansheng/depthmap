#ifndef _DEPTHMAP_H_
#define _DEPTHMAP_H_

#include <opencv2/opencv.hpp>
#include "oflow.h"


class depthmap
{
public:
    //parameter
    int lv_f, lv_l, maxiter, miniter, patchsz, patnorm, costfct, tv_innerit, tv_solverit, verbosity;
    float mindprate, mindrrate, minimgerr, poverl, tv_alpha, tv_gamma, tv_delta, tv_sor;
    bool usefbcon, usetvref;
    char* outfile;
    int rpyrtype,nochannels,incoltype;
    depthmap(int rpyrtype,int nochannels,int incoltype);
    cv::Mat get_depth(cv::Mat& input1,cv::Mat& input2);
    //cv::Mat update_depth(cv::Mat& bg_depth,std::vector<cv::Rect> result,cv::Mat& frame,cv::Mat& frame2);
    cv::Mat update_depth_robust(cv::Mat& depth_map,cv::Mat mask); 
    cv::Mat init_depth(cv::Mat& init1,cv::Mat& init2);
    void SavePFMFile(cv::Mat& img, const char* filename);
    void SaveFlowFile(cv::Mat& img, const char* filename);
    //void update_mat(Mat input1,Mat input2);

private:
    void ConstructImgPyramide(const cv::Mat & img_ao_fmat,
        cv::Mat * img_ao_fmat_pyr,
        cv::Mat * img_ao_dx_fmat_pyr,
        cv::Mat * img_ao_dy_fmat_pyr,
        const float ** img_ao_pyr,
        const float ** img_ao_dx_pyr, 
        const float ** img_ao_dy_pyr, 
        const int lv_f, 
        const int lv_l, 
        const int rpyrtype, 
        const bool getgrad, 
        const int imgpadding, 
        const int padw, 
        const int padh);
    int AutoFirstScaleSelect(int imgwidth, int fratio, int patchsize);

};

#endif
