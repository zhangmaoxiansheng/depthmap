#include "depthmap.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
using namespace std;
using namespace cv;

depthmap::depthmap(int rpyrtypei,int nochannelsi,int incoltypei)
{
    rpyrtype = rpyrtypei;
    nochannels = nochannelsi;
    incoltype = incoltypei;
    mindprate = 0.05; mindrrate = 0.95; minimgerr = 0.0;    
    usefbcon = 0; patnorm = 1; costfct = 0; 
    tv_alpha = 10.0; tv_gamma = 10.0; tv_delta = 5.0;
    tv_innerit = 1; tv_solverit = 3; tv_sor = 1.6;
    lv_f = 5;
    lv_l = 3;
    maxiter = 12;
    miniter = 12;
    mindprate = 0.05;
    mindrrate = 0.95;
    minimgerr = 0;
    patchsz = 8;
    poverl = 0.40;
    usefbcon = 0;
    patnorm = 1;
    costfct = 0;
    usetvref = 1;
    tv_alpha = 10;
    tv_gamma = 10;
    tv_delta = 5;
    tv_innerit = 1;
    tv_solverit = 3;
    tv_sor = 1.6;    
    verbosity = 0;

}
void depthmap::ConstructImgPyramide(const cv::Mat & img_ao_fmat, cv::Mat * img_ao_fmat_pyr, cv::Mat * img_ao_dx_fmat_pyr, cv::Mat * img_ao_dy_fmat_pyr, const float ** img_ao_pyr, const float ** img_ao_dx_pyr, const float ** img_ao_dy_pyr, const int lv_f, const int lv_l, const int rpyrtype, const bool getgrad, const int imgpadding, const int padw, const int padh)
{
    for (int i=0; i<=lv_f; ++i)  // Construct image and gradient pyramides
    {
        if (i==0) // At finest scale: copy directly, for all other: downscale previous scale by .5
        {
            #if (SELECTCHANNEL==1 | SELECTCHANNEL==3)  // use RGB or intensity image directly
            img_ao_fmat_pyr[i] = img_ao_fmat.clone();
            #elif (SELECTCHANNEL==2)   // use gradient magnitude image as input
            cv::Mat dx,dy,dx2,dy2,dmag;
            cv::Sobel( img_ao_fmat, dx, CV_32F, 1, 0, 3, 1/8.0, 0, cv::BORDER_DEFAULT );
            cv::Sobel( img_ao_fmat, dy, CV_32F, 0, 1, 3, 1/8.0, 0, cv::BORDER_DEFAULT );
            dx2 = dx.mul(dx);
            dy2 = dy.mul(dy);
            dmag = dx2+dy2;
            cv::sqrt(dmag,dmag);
            img_ao_fmat_pyr[i] = dmag.clone();
            #endif
        }
        else
        cv::resize(img_ao_fmat_pyr[i-1], img_ao_fmat_pyr[i], cv::Size(), .5, .5, cv::INTER_LINEAR);
	    
        img_ao_fmat_pyr[i].convertTo(img_ao_fmat_pyr[i], rpyrtype);
	
        if ( getgrad ) 
        {
            cv::Sobel( img_ao_fmat_pyr[i], img_ao_dx_fmat_pyr[i], CV_32F, 1, 0, 3, 1/8.0, 0, cv::BORDER_DEFAULT );
            cv::Sobel( img_ao_fmat_pyr[i], img_ao_dy_fmat_pyr[i], CV_32F, 0, 1, 3, 1/8.0, 0, cv::BORDER_DEFAULT );
            img_ao_dx_fmat_pyr[i].convertTo(img_ao_dx_fmat_pyr[i], CV_32F);
            img_ao_dy_fmat_pyr[i].convertTo(img_ao_dy_fmat_pyr[i], CV_32F);
        }
    }
    
    // pad images
    for (int i=0; i<=lv_f; ++i)  // Construct image and gradient pyramides
    {
        cv::copyMakeBorder(img_ao_fmat_pyr[i],img_ao_fmat_pyr[i],imgpadding,imgpadding,imgpadding,imgpadding,cv::BORDER_REPLICATE);  // Replicate border for image padding
        img_ao_pyr[i] = (float*)img_ao_fmat_pyr[i].data;

        if ( getgrad ) 
        {
            cv::copyMakeBorder(img_ao_dx_fmat_pyr[i],img_ao_dx_fmat_pyr[i],imgpadding,imgpadding,imgpadding,imgpadding,cv::BORDER_CONSTANT , 0); // Zero padding for gradients
            cv::copyMakeBorder(img_ao_dy_fmat_pyr[i],img_ao_dy_fmat_pyr[i],imgpadding,imgpadding,imgpadding,imgpadding,cv::BORDER_CONSTANT , 0);

            img_ao_dx_pyr[i] = (float*)img_ao_dx_fmat_pyr[i].data;
            img_ao_dy_pyr[i] = (float*)img_ao_dy_fmat_pyr[i].data;      
        }
    }
}

void depthmap::SaveFlowFile(cv::Mat& img, const char* filename)
{
    cv::Size szt = img.size();
    int width = szt.width, height = szt.height;
    int nc = img.channels();
    float tmp[nc];

    FILE *stream = fopen(filename, "wb");
    if (stream == 0)
    cout << "WriteFile: could not open file" << endl;

    // write the header
    fprintf(stream, "PIEH");
    if ((int)fwrite(&width,  sizeof(int),   1, stream) != 1 ||
        (int)fwrite(&height, sizeof(int),   1, stream) != 1)
    cout << "WriteFile: problem writing header" << endl;

    for (int y = 0; y < height; y++) 
    {
    for (int x = 0; x < width; x++) 
    {
        if (nc==1) // depth
        tmp[0] = img.at<float>(y,x);
        else if (nc==2) // Optical Flow
        {
        tmp[0] = img.at<cv::Vec2f>(y,x)[0];
        tmp[1] = img.at<cv::Vec2f>(y,x)[1];
        }
        else if (nc==4) // Scene Flow
        {
            tmp[0] = img.at<cv::Vec4f>(y,x)[0];
            tmp[1] = img.at<cv::Vec4f>(y,x)[1];
            tmp[2] = img.at<cv::Vec4f>(y,x)[2];
            tmp[3] = img.at<cv::Vec4f>(y,x)[3];
        }	  

        if ((int)fwrite(tmp, sizeof(float), nc, stream) != nc)
        cout << "WriteFile: problem writing data" << endl;         
    }
  }
  fclose(stream);
}
void depthmap::SavePFMFile(cv::Mat& img, const char* filename)
{
  cv::Size szt = img.size();
  
  FILE *stream = fopen(filename, "wb");
  if (stream == 0)
    cout << "WriteFile: could not open file" << endl;

  // write the header
  fprintf(stream, "Pf\n%d %d\n%f\n", szt.width, szt.height, (float)-1.0f);    
  
  for (int y = szt.height-1; y >= 0 ; --y) 
  {
    for (int x = 0; x < szt.width; ++x) 
    {
      float tmp = -img.at<float>(y,x);
      if ((int)fwrite(&tmp, sizeof(float), 1, stream) != 1)
        cout << "WriteFile: problem writing data" << endl;         
    }
  }  
  fclose(stream);
}
int depthmap::AutoFirstScaleSelect(int imgwidth, int fratio, int patchsize)
{
    return std::max(0,(int)std::floor(log2((2.0f*(float)imgwidth) / ((float)fratio * (float)patchsize))));
}

Mat depthmap::get_depth(Mat& input1,Mat& input2)
{
  
  cv::Mat img_ao_mat = input1; // Read the file
  cv::Mat img_bo_mat = input2;   // Read the file
  
  //cv::Mat temp;   
  cv::Mat img_ao_fmat, img_bo_fmat;
  cv::Size sz = img_ao_mat.size();
  int width_org = sz.width;   // unpadded original image size
  int height_org = sz.height;  // unpadded original image size
  //--------------------------------------------------preapre------------------------------------------------------------------------------------
  
  // *** Pad image such that width and height are restless divisible on all scales (except last)
  int padw=0, padh=0;
  int scfct = pow(2,lv_f); // enforce restless division by this number on coarsest scale
  //if (hasinfile) scfct = pow(2,lv_f+1); // if initialization file is given, make sure that size is restless divisible by 2^(lv_f+1) !
  int div = sz.width % scfct;
  if (div>0) padw = scfct - div;
  div = sz.height % scfct;
  if (div>0) padh = scfct - div;          
  if (padh>0 || padw>0)
  {
    cv::copyMakeBorder(img_ao_mat,img_ao_mat,floor((float)padh/2.0f),ceil((float)padh/2.0f),floor((float)padw/2.0f),ceil((float)padw/2.0f),cv::BORDER_REPLICATE);
    cv::copyMakeBorder(img_bo_mat,img_bo_mat,floor((float)padh/2.0f),ceil((float)padh/2.0f),floor((float)padw/2.0f),ceil((float)padw/2.0f),cv::BORDER_REPLICATE);
  }
  sz = img_ao_mat.size();  // padded image size, ensures divisibility by 2 on all scales (except last)
  
  img_ao_mat.convertTo(img_ao_fmat, CV_32F); // convert to float
  img_bo_mat.convertTo(img_bo_fmat, CV_32F);
  
  const float* img_ao_pyr[lv_f+1];
  const float* img_bo_pyr[lv_f+1];
  const float* img_ao_dx_pyr[lv_f+1];
  const float* img_ao_dy_pyr[lv_f+1];
  const float* img_bo_dx_pyr[lv_f+1];
  const float* img_bo_dy_pyr[lv_f+1];
  
  cv::Mat img_ao_fmat_pyr[lv_f+1];
  cv::Mat img_bo_fmat_pyr[lv_f+1];
  cv::Mat img_ao_dx_fmat_pyr[lv_f+1];
  cv::Mat img_ao_dy_fmat_pyr[lv_f+1];
  cv::Mat img_bo_dx_fmat_pyr[lv_f+1];
  cv::Mat img_bo_dy_fmat_pyr[lv_f+1];
  
  ConstructImgPyramide(img_ao_fmat, img_ao_fmat_pyr, img_ao_dx_fmat_pyr, img_ao_dy_fmat_pyr, img_ao_pyr, img_ao_dx_pyr, img_ao_dy_pyr, lv_f, lv_l, rpyrtype, 1, patchsz, padw, padh);
  ConstructImgPyramide(img_bo_fmat, img_bo_fmat_pyr, img_bo_dx_fmat_pyr, img_bo_dy_fmat_pyr, img_bo_pyr, img_bo_dx_pyr, img_bo_dy_pyr, lv_f, lv_l, rpyrtype, 1, patchsz, padw, padh);
  
  //  *** Run main optical flow / depth algorithm
  float sc_fct = pow(2,lv_l);
  
  cv::Mat flowout(sz.height / sc_fct , sz.width / sc_fct, CV_32FC1); // Depth
    
  
  OFC::OFClass ofc(img_ao_pyr, img_ao_dx_pyr, img_ao_dy_pyr, 
                    img_bo_pyr, img_bo_dx_pyr, img_bo_dy_pyr, 
                    patchsz,  // extra image padding to avoid border violation check
                    (float*)flowout.data,   // pointer to n-band output float array
                    nullptr,  // pointer to n-band input float array of size of first (coarsest) scale, pass as nullptr to disable
                    sz.width, sz.height, 
                    lv_f, lv_l, maxiter, miniter, mindprate, mindrrate, minimgerr, patchsz, poverl, 
                    usefbcon, costfct, nochannels, patnorm, 
                    usetvref, tv_alpha, tv_gamma, tv_delta, tv_innerit, tv_solverit, tv_sor,
                    verbosity);    

  // *** Resize to original scale, if not run to finest level
  if (lv_l != 0)
  {
    flowout *= sc_fct;
    cv::resize(flowout, flowout, cv::Size(), sc_fct, sc_fct , cv::INTER_LINEAR);
  }
  
  // If image was padded, remove padding before saving to file
  flowout = flowout(cv::Rect((int)floor((float)padw/2.0f),(int)floor((float)padh/2.0f),width_org,height_org));
  flowout = cv::abs(flowout);
  return flowout;
}

Mat depthmap::update_depth(Mat& bg_depth,vector<Rect> result,Mat& frame,Mat& frame2)
{
    //Mat depth_map = get_depth(frame,frame2);
    //int64 start = cv::getTickCount();
    Mat depth_map;
    bg_depth.copyTo(depth_map);
    
    Mat all_depth = get_depth(frame,frame2);
    all_depth = all_depth * 4;
    cv::resize(all_depth,all_depth,Size(4000,3000));
    for(size_t i = 0; i <result.size();i++)
    {
        Rect temp ;
        temp.x = result[i].x * 2;
		temp.y = result[i].y * 2;
	 	temp.width = result[i].width * 2;
	 	temp.height = result[i].height * 2;
        Mat temp_depth = all_depth(temp);
        temp_depth.copyTo(depth_map(temp));
        
    }
    // double fps = cv::getTickFrequency() / (cv::getTickCount() - start);
    // std::cout << "depth map all time : " << 1000/fps << std::endl;
    return depth_map;
    //return all_depth;
}

Mat depthmap::init_depth(Mat& input1,Mat& input2)
{
  
  cv::Mat img_ao_mat = input1; // Read the file
  cv::Mat img_bo_mat = input2;   // Read the file
  
  //cv::Mat temp;   
  cv::Mat img_ao_fmat, img_bo_fmat;
  cv::Size sz = img_ao_mat.size();
  int width_org = sz.width;   // unpadded original image size
  int height_org = sz.height;  // unpadded original image size

  int fratio = 5; // For automatic selection of coarsest scale: 1/fratio * width = maximum expected motion magnitude in image. Set lower to restrict search space.
    
    int sel_oppoint = 4; // Default operating point
      
    switch (sel_oppoint)
    {
      case 1:
        patchsz = 8; poverl = 0.3; 
        lv_f = AutoFirstScaleSelect(width_org, fratio, patchsz);
        lv_l = std::max(lv_f-2,0); maxiter = 16; miniter = 16; 
        usetvref = 0; 
        break;
      case 3:
        patchsz = 12; poverl = 0.75; 
        lv_f = AutoFirstScaleSelect(width_org, fratio, patchsz);
        lv_l = std::max(lv_f-4,0); maxiter = 16; miniter = 16; 
        usetvref = 1; 
        break;
      case 4:
        patchsz = 12; poverl = 0.75; 
        lv_f = AutoFirstScaleSelect(width_org, fratio, patchsz);
        lv_l = std::max(lv_f-5,0); maxiter = 128; miniter = 128; 
        usetvref = 1; 
        break;        
      case 2:
      default:
        patchsz = 8; poverl = 0.4; 
        lv_f = AutoFirstScaleSelect(width_org, fratio, patchsz);
        lv_l = std::max(lv_f-2,0); maxiter = 12; miniter = 12; 
        usetvref = 1; 
        break;

    }
  //--------------------------------------------------preapre------------------------------------------------------------------------------------
  
  // *** Pad image such that width and height are restless divisible on all scales (except last)
  int padw=0, padh=0;
  int scfct = pow(2,lv_f); // enforce restless division by this number on coarsest scale
  //if (hasinfile) scfct = pow(2,lv_f+1); // if initialization file is given, make sure that size is restless divisible by 2^(lv_f+1) !
  int div = sz.width % scfct;
  if (div>0) padw = scfct - div;
  div = sz.height % scfct;
  if (div>0) padh = scfct - div;          
  if (padh>0 || padw>0)
  {
    cv::copyMakeBorder(img_ao_mat,img_ao_mat,floor((float)padh/2.0f),ceil((float)padh/2.0f),floor((float)padw/2.0f),ceil((float)padw/2.0f),cv::BORDER_REPLICATE);
    cv::copyMakeBorder(img_bo_mat,img_bo_mat,floor((float)padh/2.0f),ceil((float)padh/2.0f),floor((float)padw/2.0f),ceil((float)padw/2.0f),cv::BORDER_REPLICATE);
  }
  sz = img_ao_mat.size();  // padded image size, ensures divisibility by 2 on all scales (except last)
  
  img_ao_mat.convertTo(img_ao_fmat, CV_32F); // convert to float
  img_bo_mat.convertTo(img_bo_fmat, CV_32F);
  
  const float* img_ao_pyr[lv_f+1];
  const float* img_bo_pyr[lv_f+1];
  const float* img_ao_dx_pyr[lv_f+1];
  const float* img_ao_dy_pyr[lv_f+1];
  const float* img_bo_dx_pyr[lv_f+1];
  const float* img_bo_dy_pyr[lv_f+1];
  
  cv::Mat img_ao_fmat_pyr[lv_f+1];
  cv::Mat img_bo_fmat_pyr[lv_f+1];
  cv::Mat img_ao_dx_fmat_pyr[lv_f+1];
  cv::Mat img_ao_dy_fmat_pyr[lv_f+1];
  cv::Mat img_bo_dx_fmat_pyr[lv_f+1];
  cv::Mat img_bo_dy_fmat_pyr[lv_f+1];
  
  ConstructImgPyramide(img_ao_fmat, img_ao_fmat_pyr, img_ao_dx_fmat_pyr, img_ao_dy_fmat_pyr, img_ao_pyr, img_ao_dx_pyr, img_ao_dy_pyr, lv_f, lv_l, rpyrtype, 1, patchsz, padw, padh);
  ConstructImgPyramide(img_bo_fmat, img_bo_fmat_pyr, img_bo_dx_fmat_pyr, img_bo_dy_fmat_pyr, img_bo_pyr, img_bo_dx_pyr, img_bo_dy_pyr, lv_f, lv_l, rpyrtype, 1, patchsz, padw, padh);
  
  //  *** Run main optical flow / depth algorithm
  float sc_fct = pow(2,lv_l);

  cv::Mat flowout(sz.height / sc_fct , sz.width / sc_fct, CV_32FC1); // Depth
  
  
  OFC::OFClass ofc(img_ao_pyr, img_ao_dx_pyr, img_ao_dy_pyr, 
                    img_bo_pyr, img_bo_dx_pyr, img_bo_dy_pyr, 
                    patchsz,  // extra image padding to avoid border violation check
                    (float*)flowout.data,   // pointer to n-band output float array
                    nullptr,  // pointer to n-band input float array of size of first (coarsest) scale, pass as nullptr to disable
                    sz.width, sz.height, 
                    lv_f, lv_l, maxiter, miniter, mindprate, mindrrate, minimgerr, patchsz, poverl, 
                    usefbcon, costfct, nochannels, patnorm, 
                    usetvref, tv_alpha, tv_gamma, tv_delta, tv_innerit, tv_solverit, tv_sor,
                    verbosity);    

  // *** Resize to original scale, if not run to finest level
  if (lv_l != 0)
  {
    flowout *= sc_fct;
    cv::resize(flowout, flowout, cv::Size(), sc_fct, sc_fct , cv::INTER_LINEAR);
  }
  
  // If image was padded, remove padding before saving to file
  flowout = flowout(cv::Rect((int)floor((float)padw/2.0f),(int)floor((float)padh/2.0f),width_org,height_org));
  flowout = cv::abs(flowout);
  return flowout;
}
