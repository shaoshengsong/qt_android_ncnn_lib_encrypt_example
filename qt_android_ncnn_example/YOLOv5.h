#ifndef YOLOv5_H
#define YOLOv5_H
#include "layer.h"
#include "net.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <float.h>
#include <stdio.h>
#include <vector>




struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

class YOLOv5
{
public:
    YOLOv5();


public:
    int init(int bgr_rgb,float prob_threshold,float nms_threshold);
    int inference(const cv::Mat& bgr, std::vector<Object>& objects);
    cv::Mat draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects);


private:
     ncnn::Net * model_;
     int bgr_rgb_=0;//输入图片的通道顺序 0:bgr   1:rgb
     float prob_threshold_ = 0.25f;
     float nms_threshold_ = 0.45f;

private:


    float intersection_area(const Object& a, const Object& b);
    void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right);
    void qsort_descent_inplace(std::vector<Object>& faceobjects);
    void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold, bool agnostic = false);
    float sigmoid(float x);
    void generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object>& objects);

};

#endif // YOLOv5_H
