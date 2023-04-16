#ifndef SE3F_TO_CV_MAT_H
#define SE3F_TO_CV_MAT_H

#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>

cv::Mat se3fToCvMat(const Sophus::SE3f &se3);

#endif  // SE3F_TO_CV_MAT_H
