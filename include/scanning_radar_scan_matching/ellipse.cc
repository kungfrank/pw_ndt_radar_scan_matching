#include <bits/stdc++.h>
#include <eigen3/Eigen/Dense>
#include <visualization_msgs/MarkerArray.h>
#include <ros/ros.h>

using visualization_msgs::Marker;
using visualization_msgs::MarkerArray;

enum class Color {
  kRed,    /**< (R, G, B) = (1.0, 0.0, 0.0) */
  kLime,   /**< (R, G, B) = (0.0, 1.0, 0.0) */
  kBlue,   /**< (R, G, B) = (0.0, 0.0, 1.0) */
  kWhite,  /**< (R, G, B, A) = (x, x, x, 0.0) */
  kBlack,  /**< (R, G, B) = (0.0, 0.0, 0.0) */
  kGray,   /**< (R, G, B) = (0.5, 0.5, 0.5) */
  kYellow, /**< (R, G, B) = (1.0, 1.0, 0.0) */
  kAqua,   /**< (R, G, B) = (0.0, 1.0, 1.0) */
  kFuchsia /**< (R, G, B) = (1.0, 0.0, 1.0) */
};

void ComputeEvalEvec(const Eigen::Matrix2d &covariance,
                     Eigen::Vector2d &evals,
                     Eigen::Matrix2d &evecs) {
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> evd;
  evd.computeDirect(covariance);
  evals = evd.eigenvalues();
  evecs = evd.eigenvectors();
}

ros::Time GetROSTime() {
  ros::Time ret;
  try {
    ret = ros::Time::now();
  } catch (const ros::Exception &ex) {
    std::cout << ex.what() << std::endl;
    std::cout << "Set stamp to ros::Time(0)" << std::endl;
    ret = ros::Time(0);
  }
  return ret;
}

std_msgs::ColorRGBA MakeColorRGBA(const Color &color, double alpha) {
  std_msgs::ColorRGBA ret;
  ret.a = alpha;
  ret.r = ret.g = ret.b = 0.;
  if (color == Color::kRed) {
    ret.r = 1.;
  } else if (color == Color::kLime) {
    ret.g = 1.;
  } else if (color == Color::kBlue) {
    ret.b = 1.;
  } else if (color == Color::kWhite) {
    ret.a = 0.;
  } else if (color == Color::kBlack) {
    // nop
  } else if (color == Color::kGray) {
    ret.r = ret.g = ret.b = 0.5;
  } else if (color == Color::kYellow) {
    ret.r = ret.g = 1.;
  } else if (color == Color::kAqua) {
    ret.g = ret.b = 1.;
  } else if (color == Color::kFuchsia) {
    ret.r = ret.b = 1.;
  }
  return ret;
}

MarkerArray JoinMarkers(const std::vector<Marker> &markers) {
  MarkerArray ret;
  int id = 0;
  auto now = GetROSTime();
  for (const auto &m : markers) {
    ret.markers.push_back(m);
    //ret.markers.back().header.stamp = now;
    ret.markers.back().id = id++;
  }
  return ret;
}

Marker MarkerOfEllipse(const Eigen::Vector2d &mean,
                       const Eigen::Matrix2d &covariance,
                       const Color &color,
                       double alpha,
                       double scale) {
  Marker ret;
  ret.header.frame_id = "res_odom";
  //ret.header.stamp = GetROSTime();
  ret.id = 0;
  ret.type = Marker::SPHERE;
  ret.action = Marker::ADD;
  ret.color = MakeColorRGBA(color, alpha);
  Eigen::Vector2d evals;
  Eigen::Matrix2d evecs;
  ComputeEvalEvec(covariance, evals, evecs);
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  R.block<2, 2>(0, 0) << evecs(0, 0), -evecs(1, 0), evecs(1, 0), evecs(0, 0);
  Eigen::Quaterniond q(R);
  ret.scale.x = scale * sqrt(evals(0));  // +- 1σ
  ret.scale.y = scale * sqrt(evals(1));  // +- 1σ
  ret.scale.z = 0.1;
  ret.pose.position.x = mean(0);
  ret.pose.position.y = mean(1);
  ret.pose.position.z = 0;
  ret.pose.orientation.x = q.x();
  ret.pose.orientation.y = q.y();
  ret.pose.orientation.z = q.z();
  ret.pose.orientation.w = q.w();
  return ret;
}
