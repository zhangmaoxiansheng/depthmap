#ifndef _GHOSTELEMER_H_
#define _GHOSTELEMER_H_
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;
class GhostElemer
{
public:
    float thresh;
    size_t size;
    int num;
    float time1;
    float time2;
    std::vector< vector<Point> > record;
    std::vector<int> table;
    GhostElemer(float thr = 4,size_t sz = 5, int num_ = 4, float t1 = 1.1,float t2 = 1.2):thresh(thr),size(sz),num(num_),time1(t1),time2(t2){}
    void ghost_dele(std::vector<cv::Rect> &res_c);
    void ghost_elem_update(std::vector<cv::Rect> &res_c);
    std::vector<Rect> Find_location(cv::Mat& img);
    std::vector<cv::Mat> Mat_res(std::vector<cv::Rect> res_c,cv::Mat frame,cv::Mat frame2);
private:
    bool ghost_range(Point a, Point b);
    void ghost_locate();
    cv::Point get_center(cv::Rect r);
    cv::Rect Large_res(cv::Rect r);
    bool Rect_Intersect(cv::Rect r1, cv::Rect r2);
    cv::Rect Rect_Join(cv::Rect r1, cv::Rect r2);
    std::vector<cv::Point> get_all_center(std::vector<cv::Rect> res_c);
};
#endif
