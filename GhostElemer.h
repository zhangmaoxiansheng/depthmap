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
    vector< vector<Point> > record;
    vector<int> table;
    GhostElemer(float thr = 4,size_t sz = 5, int num_ = 4, float t1 = 1.1,float t2 = 1.2):thresh(thr),size(sz),num(num_),time1(t1),time2(t2){}
    void ghost_dele(vector<Rect> &res_c);
    void ghost_elem_update(vector<Rect> &res_c);
    vector<Rect> Find_location(Mat img);
    vector<Mat> Mat_res(vector<Rect> res_c,Mat frame,Mat frame2);
private:
    bool ghost_range(Point a, Point b);
    void ghost_locate();
    Point get_center(Rect r);
    Rect Large_res(Rect r);
    bool Rect_Intersect(Rect r1, Rect r2);
    Rect Rect_Join(Rect r1, Rect r2);
    vector<Point> get_all_center(vector<Rect> res_c);
};
#endif
