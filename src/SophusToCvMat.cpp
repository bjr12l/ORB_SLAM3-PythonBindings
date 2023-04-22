#include "SophusToCvMat.h"
#include <opencv2/core/eigen.hpp>

cv::Mat se3fToCvMat(const Sophus::SE3f &se3) {
    Eigen::Matrix4f eigen_matrix = se3.matrix();
    cv::Mat opencv_matrix;
    cv::eigen2cv(eigen_matrix, opencv_matrix);
    return opencv_matrix;
}
