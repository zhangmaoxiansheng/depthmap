#include<iostream>
#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include "GhostElemer.h"

using namespace std;
using namespace cv;

bool GhostElemer::ghost_range(Point a, Point b)
{
	float dis = powf((a.x - b.x), 2) + powf((a.y - b.y), 2);
	//cout << dis << endl;
	return dis < powf(thresh, 2);
}
void GhostElemer::ghost_locate()
{

	vector< vector<Point> >::iterator it = record.begin();
	vector<Point> latest = record.back();
    if(!table.empty())
        table.clear();
    vector<int> table1(latest.size());
    table = table1;
	for (; it != record.end() - 1; it++)
	{
		//cout << "!!!!!!!!!!!!!!!!!!!!" << endl;
		for (size_t i = 0; i < latest.size(); i++)
		{
			for (size_t jj = 0; jj < (*it).size(); jj++)
			{
				if (ghost_range(latest[i], (*it)[jj]))
				{
					table[i]++;
					//cout << "this one:" << i << endl;
				}
			}
		}
	}
}

void GhostElemer::ghost_dele(vector<Rect> &res_c)
{
	//
	for (size_t i = 0; i < res_c.size(); )
	{
		if (table[i] >= num)
		{
			//cout << "dele this one:" << i << endl;
			//record.back().erase(record.back().begin() + i);
			res_c.erase(res_c.begin() + i);
			table.erase(table.begin() + i);
		}
		else
			i++;
	}
}
Point GhostElemer:: get_center(Rect r)
{
	Point rr;
	rr.x = r.x + r.width / 2;
	rr.y = r.y + r.height / 2;
	return rr;
}
vector<Point> GhostElemer::get_all_center(vector<Rect> res_c)
{
	vector<Point> n;
	for (size_t i = 0; i < res_c.size(); i++)
	{
		n.push_back(get_center(res_c[i]));
	}
	return n;
}
void GhostElemer::ghost_elem_update(vector<Rect> &res_c)
{
	if (record.size() < size)
	{
		cout << "init" << endl;
		record.push_back(get_all_center(res_c));//init
	}
	if (record.size() >= size)
	{
		cout << "test1" << endl;
		ghost_locate();
        ghost_dele(res_c);
		record.erase(record.begin());
		cout << "erase first" << endl;
	}

}
Rect GhostElemer::Large_res(Rect r)
{
	Rect n;
	if (r.width > 200 || r.height > 200)
	{
		n.x = r.x - (time1 - 1) / 2 * r.width;
		n.y = r.y - (time1 - 1) / 2 * r.height;
		n.width = r.width*time1;
		n.height = r.height*time1;
	}
	else
	{
		n.x = r.x - (time2 - 1) / 2 * r.width;
		n.y = r.y - (time2 - 1) / 2 * r.height;
		n.width = r.width*time2;
		n.height = r.height*time2;
	}
	return n;
}
bool GhostElemer::Rect_Intersect(Rect r1, Rect r2)
{
	int dis_x = (r1.width + r2.width) / 2;
	int dis_y = (r1.height + r2.height) / 2;
	int x = abs((r1.x + r1.width / 2) - (r2.x + r2.width / 2));
	int y = abs((r1.y + r1.height / 2) - (r2.y + r2.height / 2));
	return (x <= dis_x && y <= dis_y);
}
Rect GhostElemer::Rect_Join(Rect r1, Rect r2)
{
	Rect join;
	join.x = min(r1.x, r2.x);
	join.y = min(r1.y, r2.y);
	join.width = max((r1.x + r1.width), (r2.x + r2.width)) - join.x;
	join.height = max((r1.y + r1.height), (r2.y + r2.height)) - join.y;
	return join;
}
vector<Rect> GhostElemer::Find_location(Mat& img)
{
	vector<Rect> res_c;
	vector<vector<Point> > contours;
	cv::findContours(img, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	vector<Rect> boundRect(contours.size());
	for (size_t i = 0; i < contours.size(); i++)
	{
		boundRect[i] = Large_res(cv::boundingRect(contours[i]));//1.2
		if (boundRect[i].area() > 500)
		{
			res_c.push_back(boundRect[i]);
			for (size_t ii = 0; ii < res_c.size() - 1; ii++)
			{
				if (Rect_Intersect(res_c.back(), res_c[ii]))//
				{
					res_c.back() = Rect_Join(res_c.back(), res_c[ii]);//
					res_c.erase(res_c.begin() + ii);
					ii = -1;
					//bug 
					//
				}
			}
		}
	}
	img = cv::Mat::zeros(img.size(),CV_8UC1);//日后优化
	for(size_t j = 0; j < res_c.size();j++)
	{
		cv::rectangle(img,res_c[j],Scalar(255,255,255),-1);
	}
	return res_c;
}
vector<Mat> GhostElemer::Mat_res(vector<Rect> res_c,Mat frame,Mat frame2)
{
	vector<Mat> Mat_res;
	for(size_t i = 0; i < res_c.size();i++)
	{
		Rect temp;
		temp.x = res_c[i].x * 2;
		temp.y = res_c[i].y * 2;
		temp.width = res_c[i].width * 2;
		temp.height = res_c[i].height * 2;
		Mat temp_mat = frame(temp);
		Mat temp_mat2 = frame2(temp);
		Mat_res.push_back(temp_mat);
		Mat_res.push_back(temp_mat2);
	}
	return Mat_res;

}
