#include <ros/ros.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseWithCovariance.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_representation.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>

#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <iterator>
#include <stdlib.h>
#include <iomanip>
#include <thread>
#include <sys/types.h>
#include <dirent.h>

#include "../include/pw_ndt_radar_scan_matching/ellipse.cc"
#include "../include/pw_ndt_radar_scan_matching/pw_ndt.h"
#include "../include/pw_ndt_radar_scan_matching/ndt.h"

using namespace std;
using namespace cv;
using namespace std::chrono_literals;

//-- golbal variable --//
vector<Eigen::Matrix4f> init_guess_vec;
vector<Eigen::Matrix4f> gt_vec;
nav_msgs::Path gt_path_msg;

float begin_t = 0;
double old_delta_dis = 0;
bool init = false;  int Index = 0;
float old_vel = 0; float old_ang_vel = 0; double old_t = 0;
float curr_vel = 0; float curr_ang_vel = 0; double curr_t = 0;
pcl::PointCloud<pcl::PointXYZI>::Ptr old_in_pc (new pcl::PointCloud<pcl::PointXYZI>);
Eigen::Matrix4f curr_tf;

string directory = "DATA_PATH";
string gt_directory = "GT_PATH";
string save_directory = "SAVE_PATH";
string init_directory = "no_init_directory";
string bag_name = "no_specific_bag";

//-- NDT param --//
float intensity_filter = 0;
int image_width = 125;
int ndt2d_grid_extent_ = int(image_width/2);
float ndt2d_grid_resolution_ = 0.125;
int ndt2d_max_it_ = 300;
double ndt2d_eps_ = 0.003;
double max_step_size_ = 0.03;

//-- Debug param --//
int global_viz = 0;
int start_index = 0;
int save_res = 0;
string res_file_name = "_res.txt";

//////////////////////////////////////

float global_threshold;
float global_bias;
float grid_size1, grid_size2, grid_size3, grid_size4;
float eps1, eps2, eps3, eps4;
double max_step_size1, max_step_size2, max_step_size3, max_step_size4;
float global_mask_flag;

ros::Publisher old_pc_pub;
ros::Publisher new_pc_pub;
ros::Publisher out_pc_pub;
ros::Publisher nd_map_pub;

ros::Publisher gt_path_pub;
ros::Publisher gt_pose_pub;
ros::Publisher res_pose_pub;

ros::Publisher err_pub;

pcl::PointCloud<pcl::PointXYZI>::Ptr match_scan(new pcl::PointCloud<pcl::PointXYZI>);
sensor_msgs::PointCloud2 old_pc;
sensor_msgs::PointCloud2 new_pc;
sensor_msgs::PointCloud2 out_pc;

template <class Container>
void SplitStr(const std::string& str, Container& cont,
              char delim = ' ')
{
    std::size_t current, previous = 0;
    current = str.find(delim);
    while (current != std::string::npos) {
        cont.push_back(str.substr(previous, current - previous));
        previous = current + 1;
        current = str.find(delim, previous);
    }
    cont.push_back(str.substr(previous, current - previous));
}

pcl::PointCloud<pcl::PointXYZI>::Ptr ImageToPointCloud(Mat& image_raw, Mat& image_mask){ // raw, mask
  pcl::PointCloud<pcl::PointXYZI>::Ptr pc(new pcl::PointCloud<pcl::PointXYZI>);
  int mid = (image_raw.rows - 1)/2;
  pcl::PointXYZI point;
  //cout << "global_mask_flag: " << global_mask_flag << endl;
  if(global_mask_flag == 1){
    //cout << "use mask" << endl;
    for(int i=0; i<image_raw.rows; i++){
      for(int j=0; j<image_raw.cols; j++){
        float point_value = (int)image_raw.at<uchar>(i,j);
        float check_point = (int)image_mask.at<uchar>(i,j);
        point.x = -(i - mid)*ndt2d_grid_resolution_;
        point.y = -(j - mid)*ndt2d_grid_resolution_;
        point.z = 0;
        point.intensity = point_value;
        if(check_point > 0 && point_value > global_threshold){
          pc->points.push_back(point);
        }
      }
    }
  }
  else{
    //cout << "no mask" << endl;
    for(int i=0; i<image_raw.rows; i++){
      for(int j=0; j<image_raw.cols; j++){
        float point_value = (int)image_raw.at<uchar>(i,j);
        point.x = -(i - mid)*ndt2d_grid_resolution_;
        point.y = -(j - mid)*ndt2d_grid_resolution_;
        point.z = 0;
        point.intensity = point_value;
        if(point_value > global_threshold){
          pc->points.push_back(point);
        }
      }
    }
  }

  return pc;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr ImageToPointCloud(Mat& image_raw){ // raw
  pcl::PointCloud<pcl::PointXYZI>::Ptr pc(new pcl::PointCloud<pcl::PointXYZI>);
  int mid = (image_raw.rows - 1)/2;
  pcl::PointXYZI point;
  for(int i=0; i<image_raw.rows; i++){
    for(int j=0; j<image_raw.cols; j++){
      float point_value = (int)image_raw.at<uchar>(i,j);
      point.x = -(i - mid)*ndt2d_grid_resolution_;
      point.y = -(j - mid)*ndt2d_grid_resolution_;
      point.z = 0;
      point.intensity = point_value;
      pc->points.push_back(point);
    }
  }
  return pc;
}

bool isNanInMatrix(Eigen::Matrix4f& T){
  for(int i=0; i<4; i++){
    for(int j=0; j<4; j++){
      if(T(i,j) != T(i,j) ){ // detect Nan
        return 1;
      }
    }
  }
  return 0;
}

vector<Eigen::Matrix4f> loadPoses(string file_name) {
  vector<Eigen::Matrix4f> poses;

  Eigen::Matrix4f P = Eigen::Matrix4f::Identity();
  poses.push_back(P); // id=0

  FILE *fp = fopen(file_name.c_str(),"r");
  if (!fp)
    return poses;
  while (!feof(fp)) {
    Eigen::Matrix4f P = Eigen::Matrix4f::Identity();
    if (fscanf(fp, "%f %f %f %f %f %f %f %f %f %f %f %f",
                   &P(0,0), &P(0,1), &P(0,2), &P(0,3),
                   &P(1,0), &P(1,1), &P(1,2), &P(1,3),
                   &P(2,0), &P(2,1), &P(2,2), &P(2,3) )==12) {
      poses.push_back(P);
    }
  }
  fclose(fp);
  return poses;
}

///////////////////////////////////////////////////////////////////////////
void pub_odom_msg(ros::Publisher pose_pub, Eigen::Matrix4f P, string child_frame_id);
void Callback(pcl::PointCloud<pcl::PointXYZI>::Ptr in_pc);
void read_directory(const std::string& name, std::vector<std::string>& v);
void WriteToFile(Eigen::Matrix4f& T, string& file_path);
pcl::PointCloud<pcl::PointXYZI>::Ptr make2DPointCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr pc_3d);
pcl::PointCloud<pcl::PointXYZI>::Ptr RadarPointCloudFilter(pcl::PointCloud<pcl::PointXYZI>::Ptr pc_in, float thres);
float GetDistanceFromT(Eigen::Matrix4f T);
float VelocityCal(Eigen::Matrix4f T, double delta_t);
float AngularVelocityCal(Eigen::Matrix4f T, double delta_t);
void DoPWNDT(const pcl::PointCloud<pcl::PointXYZI>::Ptr old_in_pc, const pcl::PointCloud<pcl::PointXYZI>::Ptr in_pc, pcl::PointCloud<pcl::PointXYZI>::Ptr output_cloud, float grid_step_, double eps_, double max_step_size_, float bias_,
              Eigen::Matrix4f &T, double &score, Eigen::Matrix<pcl::ndt2d::NormalDist<pcl::PointXYZI>, Eigen::Dynamic, Eigen::Dynamic> &normal_distributions_map);
nav_msgs::Path gen_path_msg(vector<Eigen::Matrix4f> pose_vec);
bool CheckAcc(Eigen::Matrix4f& T);
void PrintAcc(Eigen::Matrix4f& T);
void NDTMultiStage(const pcl::PointCloud<pcl::PointXYZI>::Ptr old_in_pc, const pcl::PointCloud<pcl::PointXYZI>::Ptr in_pc, Eigen::Matrix4f &T_final, pcl::PointCloud<pcl::PointXYZI>::Ptr &output_cloud);
void Vizsualization(bool viz,
                    const pcl::PointCloud<pcl::PointXYZI>::Ptr in_pc,
                    const pcl::PointCloud<pcl::PointXYZI>::Ptr output_cloud,
                    const pcl::PointCloud<pcl::PointXYZI>::Ptr old_in_pc);
void deleteNDMap();
void pubNDMap(Eigen::Matrix< pcl::ndt2d::NormalDist<pcl::PointXYZI> , Eigen::Dynamic, Eigen::Dynamic> normal_distributions_map,
              double scale);
///////////////////////////////////////////////////////////////////////////

pcl::PointCloud<pcl::PointXYZI>::Ptr removeInfinePoint(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud){
  pcl::PointCloud<pcl::PointXYZI>::iterator it = cloud->points.begin();
  while (it != cloud->points.end())
  {

    float x, y, z;
    x = it->x;
    y = it->y;
    z = it->z;
    if (!pcl_isfinite(x) || !pcl_isfinite(y) || !pcl_isfinite(z))
    {
      it = cloud->points.erase(it);
    }
    else
      ++it;
  }
  return cloud;
}

void OpenBag(string bag_name){
  //-- Load GT --//
  gt_vec = loadPoses(gt_directory+"/"+bag_name+"_gt.txt");
  cout << "GT file size: " << gt_vec.size() << endl;
  gt_path_msg = gen_path_msg(gt_vec);
  //-- Load init guess --//
  if (init_directory !="no_init_directory"){
    cout << init_directory+"/"+bag_name+"_res.txt" << endl;
    init_guess_vec = loadPoses(init_directory+"/"+bag_name+"_res.txt");
    cout << "Initial guess file size: " << init_guess_vec.size() << endl;
  }

  string path = directory+"/"+bag_name+"/";
  ifstream file(path + "radar_t_list"); string str;
  cout << "OpenBag \n";
  curr_tf.setIdentity();  // Init TF
  init = false;
  Index = 0;
  curr_vel = 0; curr_ang_vel = 0;
  old_delta_dis = 0;

  while (std::getline(file, str)) {
    std::vector<std::string> words;
    SplitStr(str, words);
    float ts = std::stof(words[0]);
    if(ts<0){
      continue;
    }
    if(Index<start_index){
      Index++;
      continue;
    }

    Mat image_raw;
    string imageName1 = path + words[0] + ".png";
    curr_t = stof(words[0]);
    image_raw = imread( imageName1, CV_8UC1 );

    pcl::PointCloud<pcl::PointXYZI>::Ptr pc(new pcl::PointCloud<pcl::PointXYZI>);
    pc = ImageToPointCloud(image_raw);
    Callback(pc);

    //namedWindow("waitkey", WINDOW_NORMAL );
    //waitKey(0);

    gt_path_pub.publish(gt_path_msg);
    Index++;
    pcl::PointCloud<pcl::PointXYZI>::Ptr empty_pc(new pcl::PointCloud<pcl::PointXYZI>);
    pc = empty_pc;
  }
}

void Callback(pcl::PointCloud<pcl::PointXYZI>::Ptr in_pc){
  if (!in_pc->is_dense){
      in_pc->is_dense = false;
      std::vector< int > indices;
      pcl::removeNaNFromPointCloud(*in_pc,*in_pc, indices);
    }

  if(!init){init = true;}
  else{
    cout << "-----------------------------------" << endl;
    cout << "[Index]: " << Index << " t:" << begin_t+curr_t << endl;
    if(old_delta_dis > 3.5){
      //ROS_ERROR("old_delta_dis: %f", old_delta_dis);
      cout << "{ERROR} [index] " << " t:" << begin_t+curr_t << Index << " old_delta_dis: " << old_delta_dis;
    }

    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    pcl::PointCloud<pcl::PointXYZI>::Ptr output_cloud (new pcl::PointCloud<pcl::PointXYZI>);

    //-- Use init guess --//
    if (init_directory != "no_init_directory"){
      Eigen::Matrix4f T_init(Eigen::Matrix4f::Identity());
      Eigen::Matrix4f P_new = init_guess_vec[Index];
      Eigen::Matrix4f P_old = init_guess_vec[Index-1];
      T_init = P_new.inverse() * P_old;
      cout << " Load init guess ";
      float init_curr_vel = VelocityCal(T_init, curr_t - old_t);
      if(init_curr_vel < 0.5){
        T = T_init;
        curr_vel = VelocityCal(T, curr_t - old_t);
        curr_ang_vel = AngularVelocityCal(T, curr_t - old_t);
      }
      else{
        T = T_init;
        in_pc = removeInfinePoint(in_pc);
        old_in_pc = removeInfinePoint(old_in_pc);
        NDTMultiStage(old_in_pc, in_pc, T, output_cloud);
      }
    }
    //-- Use init guess end --//
    else{
      in_pc = removeInfinePoint(in_pc);
      old_in_pc = removeInfinePoint(old_in_pc);
      NDTMultiStage(old_in_pc, in_pc, T, output_cloud);
    }
    //-- Pub pointcloud msg --//
    pcl::PointCloud<pcl::PointXYZI>::Ptr in_pc_flt (new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr old_in_pc_flt (new pcl::PointCloud<pcl::PointXYZI>);
    in_pc_flt = RadarPointCloudFilter(in_pc, 85);
    old_in_pc_flt = RadarPointCloudFilter(old_in_pc, 85);
    pcl::toROSMsg(*in_pc_flt, new_pc);
    pcl::toROSMsg(*old_in_pc_flt, old_pc);
    pcl::toROSMsg(*output_cloud, out_pc);
    old_pc.header.frame_id = "res_odom";
    new_pc.header.frame_id = "res_odom";
    out_pc.header.frame_id = "res_odom";
    old_pc_pub.publish(old_pc);
    new_pc_pub.publish(new_pc);
    out_pc_pub.publish(out_pc);
    //-- Show result --//
    cout << " [RES] ";
    PrintAcc(T);
    //-- Read GT --//
    Eigen::Matrix4f T_gt (Eigen::Matrix4f::Identity());
    Eigen::Matrix4f P_gt_new = gt_vec[Index];
    Eigen::Matrix4f P_gt_old = gt_vec[Index-1];
    T_gt = P_gt_new.inverse() * P_gt_old;
    //-- Show GT --//
    cout << " [GT] ";
    PrintAcc(T_gt);
    //-- Show error --//
    float x_gt = T_gt(0,3); float y_gt = T_gt(1,3); float theta_gt = atan2(T_gt(1,0), T_gt(0,0));
    float x_res = T(0,3); float y_res = T(1,3); float theta_res = atan2(T(1,0), T(0,0));
    cout << " [ERROR] xytheta: " << abs(x_res-x_gt) << " m "
                                 << abs(y_res-y_gt) << " m "
                                 << abs(theta_res-theta_gt)*180/3.14 << " deg" << endl;
    // Update
    curr_vel = VelocityCal(T, curr_t - old_t);
    curr_ang_vel = AngularVelocityCal(T, curr_t - old_t);
    old_delta_dis = GetDistanceFromT(T);
    curr_tf = T * curr_tf;
    Eigen::Matrix4f curr_tf_inv =  curr_tf.inverse();

    //-- Write result to file --//
    string file_path = save_directory + "/" + bag_name + res_file_name;
    if (save_res != 0){
      WriteToFile(curr_tf_inv, file_path);
    }
    //-- Pub odom msg --//
    pub_odom_msg(gt_pose_pub, P_gt_new, "gt_odom");
    pub_odom_msg(res_pose_pub, curr_tf_inv, "res_odom");

    //-- Pub error msg --//
    nav_msgs::Odometry err_msg;
    err_msg.header.seq = Index;
    err_msg.pose.pose.position.x = abs(x_res-x_gt);
    err_msg.pose.pose.position.y = abs(y_res-y_gt);
    err_msg.pose.pose.position.z = abs(theta_res-theta_gt)*180/3.14;
    err_pub.publish(err_msg);

    //if(CheckAcc(T))
    //  Vizsualization(1, in_pc_flt, output_cloud, old_in_pc_flt);
    Vizsualization(global_viz, in_pc, output_cloud, old_in_pc);
  }
  // Update
  old_in_pc = in_pc;
  old_vel = curr_vel;
  old_ang_vel = curr_ang_vel;
  old_t = curr_t;
}
nav_msgs::Path gen_path_msg(vector<Eigen::Matrix4f> pose_vec){
  nav_msgs::Path path_msg;
  path_msg.header.frame_id = "origin";
  geometry_msgs::PoseStamped p;
  for (int i=0; i<pose_vec.size(); i++){
    Eigen::Matrix4f P;
    P = pose_vec[i];
    tf::Vector3 origin;
    origin.setValue(static_cast<double>(P(0,3)),static_cast<double>(P(1,3)),static_cast<double>(P(2,3)));
    tf::Matrix3x3 tf3d;
    tf3d.setValue(static_cast<double>(P(0,0)), static_cast<double>(P(0,1)), static_cast<double>(P(0,2)),
          static_cast<double>(P(1,0)), static_cast<double>(P(1,1)), static_cast<double>(P(1,2)),
          static_cast<double>(P(2,0)), static_cast<double>(P(2,1)), static_cast<double>(P(2,2)));
    tf::Quaternion tfqt;
    tf3d.getRotation(tfqt);
    tf::Transform transform;
    transform.setOrigin(origin);
    transform.setRotation(tfqt);

    tf::poseTFToMsg(transform, p.pose);
    path_msg.poses.push_back(p);
  }
  return path_msg;
}


void pub_odom_msg(ros::Publisher pose_pub, Eigen::Matrix4f P, string child_frame_id){
  tf::Vector3 origin;
  origin.setValue(static_cast<double>(P(0,3)),static_cast<double>(P(1,3)),static_cast<double>(P(2,3)));
  tf::Matrix3x3 tf3d;
  tf3d.setValue(static_cast<double>(P(0,0)), static_cast<double>(P(0,1)), static_cast<double>(P(0,2)),
        static_cast<double>(P(1,0)), static_cast<double>(P(1,1)), static_cast<double>(P(1,2)),
        static_cast<double>(P(2,0)), static_cast<double>(P(2,1)), static_cast<double>(P(2,2)));
  tf::Quaternion tfqt;
  tf3d.getRotation(tfqt);
  tf::Transform transform;
  transform.setOrigin(origin);
  transform.setRotation(tfqt);

  nav_msgs::Odometry  odom_msg;
  //odom_msg.header.stamp = input->header.stamp;
  odom_msg.header.frame_id = "/origin";
  odom_msg.child_frame_id = "/origin";
  tf::poseTFToMsg(transform, odom_msg.pose.pose);
  pose_pub.publish(odom_msg);

  static tf2_ros::TransformBroadcaster br;
  geometry_msgs::TransformStamped transformStamped;
  transformStamped.header.frame_id = "origin";
  transformStamped.child_frame_id = child_frame_id;
  transformStamped.transform.translation.x = odom_msg.pose.pose.position.x;
  transformStamped.transform.translation.y = odom_msg.pose.pose.position.y;
  transformStamped.transform.translation.z = odom_msg.pose.pose.position.z;
  transformStamped.transform.rotation.x = odom_msg.pose.pose.orientation.x;
  transformStamped.transform.rotation.y = odom_msg.pose.pose.orientation.y;
  transformStamped.transform.rotation.z = odom_msg.pose.pose.orientation.z;
  transformStamped.transform.rotation.w = odom_msg.pose.pose.orientation.w;
  br.sendTransform(transformStamped);

}


void NDTMultiStage(const pcl::PointCloud<pcl::PointXYZI>::Ptr old_in_pc, const pcl::PointCloud<pcl::PointXYZI>::Ptr in_pc, Eigen::Matrix4f &T, pcl::PointCloud<pcl::PointXYZI>::Ptr &output_cloud){
  double score;

  pcl::PointCloud<pcl::PointXYZI>::Ptr in_pc_flt (new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr old_in_pc_flt (new pcl::PointCloud<pcl::PointXYZI>);
  in_pc_flt = RadarPointCloudFilter(in_pc, global_threshold);
  old_in_pc_flt = RadarPointCloudFilter(old_in_pc, global_threshold);

  Eigen::Matrix< pcl::ndt2d::NormalDist<pcl::PointXYZI> , Eigen::Dynamic, Eigen::Dynamic> normal_distributions_map1, normal_distributions_map2, normal_distributions_map3, normal_distributions_map4;

  if (curr_vel<0.5){
    cout << "[[[ low speed mode ]]]" << endl;
    DoPWNDT(old_in_pc_flt, in_pc_flt, output_cloud, 1.5, 0.006, 0.0065, 0, T, score, normal_distributions_map3);
  }
  else{
    DoPWNDT(old_in_pc_flt, in_pc_flt, output_cloud, grid_size1, eps1, max_step_size1, 0, T, score, normal_distributions_map1);
    DoPWNDT(old_in_pc_flt, in_pc_flt, output_cloud, grid_size2, eps2, max_step_size2, global_bias, T, score, normal_distributions_map2);
    in_pc_flt = RadarPointCloudFilter(in_pc, 85); old_in_pc_flt = RadarPointCloudFilter(old_in_pc, 85);
    DoPWNDT(old_in_pc_flt, in_pc_flt, output_cloud, grid_size3, eps3, max_step_size3, 85, T, score, normal_distributions_map3);
  }

  deleteNDMap();
  pubNDMap(normal_distributions_map3, 1.5);

}


int main(int argc, char** argv)
{
  ros::init (argc, argv, "odometry_intensity_test");
  ros::NodeHandle nh("~");

  old_pc_pub = nh.advertise<sensor_msgs::PointCloud2>("vis/old_pc", 10);
  new_pc_pub = nh.advertise<sensor_msgs::PointCloud2>("vis/new_pc", 10);
  out_pc_pub = nh.advertise<sensor_msgs::PointCloud2>("vis/out_pc", 10);
  nd_map_pub = nh.advertise<visualization_msgs::MarkerArray>("vis/ndmap", 10);
  gt_path_pub = nh.advertise<nav_msgs::Path>("vis/gt_path", 10);
  gt_pose_pub = nh.advertise<nav_msgs::Odometry>("vis/gt_pose", 10);
  res_pose_pub = nh.advertise<nav_msgs::Odometry>("vis/res_pose", 10);
  err_pub = nh.advertise<nav_msgs::Odometry>("err", 10);

  if (!nh.getParam("directory", directory)) ROS_WARN("[%s] Failed to get param 'directory', use default setting: %s", ros::this_node::getName().c_str(), directory);
  if (!nh.getParam("gt_directory", gt_directory)) ROS_WARN("[%s] Failed to get param 'gt_directory', use default setting: %s", ros::this_node::getName().c_str(), gt_directory);
  if (!nh.getParam("save_directory", save_directory)) ROS_WARN("[%s] Failed to get param 'save_directory', use default setting: %s", ros::this_node::getName().c_str(), save_directory);
  if (!nh.getParam("init_directory", init_directory)) ROS_WARN("[%s] Failed to get param 'init_directory', use default setting: %s", ros::this_node::getName().c_str(), init_directory);
  if (!nh.getParam("bag_name", bag_name)) ROS_WARN("[%s] Failed to get param 'bag_name', use default setting: %s", ros::this_node::getName().c_str(), bag_name);
  if (!nh.getParam("res_file_name", res_file_name)) ROS_WARN("[%s] Failed to get param 'res_file_name', use default setting: %s", ros::this_node::getName().c_str(), res_file_name);
  if (!nh.getParam("global_viz", global_viz)) ROS_WARN("[%s] Failed to get param 'global_viz', use default setting: %f", ros::this_node::getName().c_str(), global_viz);
  if (!nh.getParam("save_res", save_res)) ROS_WARN("[%s] Failed to get param 'save_res', use default setting: %f", ros::this_node::getName().c_str(), save_res);
  if (!nh.getParam("start_index", start_index)) ROS_WARN("[%s] Failed to get param 'start_index', use default setting: %f", ros::this_node::getName().c_str(), start_index);
  if (!nh.getParam("global_threshold", global_threshold)) ROS_WARN("[%s] Failed to get param 'global_threshold', use default setting: %f", ros::this_node::getName().c_str(), global_threshold);
  if (!nh.getParam("global_bias", global_bias)) ROS_WARN("[%s] Failed to get param 'global_bias', use default setting: %f", ros::this_node::getName().c_str(), global_bias);
  if (!nh.getParam("global_mask_flag", global_mask_flag)) ROS_WARN("[%s] Failed to get param 'global_mask_flag', use default setting: %f", ros::this_node::getName().c_str(), global_mask_flag);

  if (!nh.getParam("grid_size1", grid_size1)) ROS_WARN("[%s] Failed to get param 'grid_size1', use default setting: %f", ros::this_node::getName().c_str(), grid_size1);
  if (!nh.getParam("grid_size2", grid_size2)) ROS_WARN("[%s] Failed to get param 'grid_size2', use default setting: %f", ros::this_node::getName().c_str(), grid_size2);
  if (!nh.getParam("grid_size3", grid_size3)) ROS_WARN("[%s] Failed to get param 'grid_size3', use default setting: %f", ros::this_node::getName().c_str(), grid_size3);
  if (!nh.getParam("eps1", eps1)) ROS_WARN("[%s] Failed to get param 'eps1', use default setting: %f", ros::this_node::getName().c_str(), eps1);
  if (!nh.getParam("eps2", eps2)) ROS_WARN("[%s] Failed to get param 'eps2', use default setting: %f", ros::this_node::getName().c_str(), eps2);
  if (!nh.getParam("eps3", eps3)) ROS_WARN("[%s] Failed to get param 'eps3', use default setting: %f", ros::this_node::getName().c_str(), eps3);
  if (!nh.getParam("max_step_size1", max_step_size1)) ROS_WARN("[%s] Failed to get param 'max_step_size1', use default setting: %f", ros::this_node::getName().c_str(), max_step_size1);
  if (!nh.getParam("max_step_size2", max_step_size2)) ROS_WARN("[%s] Failed to get param 'max_step_size2', use default setting: %f", ros::this_node::getName().c_str(), max_step_size2);
  if (!nh.getParam("max_step_size3", max_step_size3)) ROS_WARN("[%s] Failed to get param 'max_step_size3', use default setting: %f", ros::this_node::getName().c_str(), max_step_size3);

  //-- Load input --//
  std::vector<std::string> folder_vec;
  read_directory(directory, folder_vec);

  if (bag_name != "no_specific_bag"){
    cout << "run specific bag: " << bag_name << endl;
    OpenBag(bag_name);
  }
  else{
    for(int i=0;i<folder_vec.size();i++){
      if(folder_vec[i].size() > 20){
        cout << i << ": " <<  folder_vec[i] << endl;
        bag_name = folder_vec[i];
        OpenBag(bag_name);
      }
    }
  }

  return 0;
}

void read_directory(const std::string& name, std::vector<std::string>& v)
{
    DIR* dirp = opendir(name.c_str());
    struct dirent * dp;
    while ((dp = readdir(dirp)) != NULL) {
        v.push_back(dp->d_name);
    }
    closedir(dirp);
}

void WriteToFile(Eigen::Matrix4f& T, string& file_path){
  FILE *fp;
  fp = fopen(file_path.c_str(),"a");
  fprintf(fp,"%f %f %f %f %f %f %f %f %f %f %f %f\n",
          T(0,0),T(0,1),0.0 ,T(0,3),
          T(1,0),T(1,1),0.0 ,T(1,3),
          0.0   ,0.0   ,1.0 ,T(2,3));
  fclose(fp);
}

pcl::PointCloud<pcl::PointXYZI>::Ptr make2DPointCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr pc_3d){
  pcl::PointCloud<pcl::PointXYZI>::Ptr pc_2d (new pcl::PointCloud<pcl::PointXYZI>);
  pcl::copyPointCloud(*pc_3d, *pc_2d);
  for(int i=0 ; i<pc_2d->size(); i++){
    pc_2d->points[i].z = 0;
  }
  return pc_2d;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr RadarPointCloudFilter(pcl::PointCloud<pcl::PointXYZI>::Ptr pc_in, float thres){
  pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out (new pcl::PointCloud<pcl::PointXYZI>);
  for(int i=0 ; i<pc_in->size(); i++){
    if(pc_in->points[i].intensity > thres){
      pc_out->push_back(pc_in->points[i]);
    }
  }
  return pc_out;
}

float GetDistanceFromT(Eigen::Matrix4f T){
  return sqrt( pow(T(0,3),2) + pow(T(1,3),2) );
}

float VelocityCal(Eigen::Matrix4f T, double delta_t){
  return sqrt(pow(T(0,3),2) + pow(T(1,3),2)) / delta_t;
}

float AngularVelocityCal(Eigen::Matrix4f T, double delta_t){
  float sy = sqrt(T(0,0) * T(0,0) +  T(1,0) * T(1,0) );
  bool singular = sy < 1e-6;
  float delta_yaw = 0;
  if(!singular){
    delta_yaw = atan2(T(1,0), T(0,0));
  }
  else{
    delta_yaw = 0;
  }
  return -(delta_yaw/M_PI*180) / delta_t; // in degree
}

void DoPWNDT(const pcl::PointCloud<pcl::PointXYZI>::Ptr old_in_pc, const pcl::PointCloud<pcl::PointXYZI>::Ptr in_pc, pcl::PointCloud<pcl::PointXYZI>::Ptr output_cloud, float grid_step_, double eps_, double max_step_size_, float bias_,
              Eigen::Matrix4f &T, double &score, Eigen::Matrix< pcl::ndt2d::NormalDist<pcl::PointXYZI> , Eigen::Dynamic, Eigen::Dynamic>& normal_distributions_map){

  bool use_pw_ndt = 1;
  pcl::PWNormalDistributionsTransform2D<pcl::PointXYZI, pcl::PointXYZI> pw_ndt;

  if (use_pw_ndt){
    pw_ndt.setMaximumIterations (ndt2d_max_it_);
    pw_ndt.setTransformationEpsilon (eps_);
    Eigen::Vector2f grid_center;	grid_center << 0, 0;
    pw_ndt.setGridCentre (grid_center);
    Eigen::Vector2f grid_extent;	grid_extent << ndt2d_grid_extent_, ndt2d_grid_extent_;
    pw_ndt.setGridExtent (grid_extent);
    Eigen::Vector2f grid_step;	grid_step << grid_step_, grid_step_;
    pw_ndt.setGridStep (grid_step);
    pw_ndt.setMaxStepSize(max_step_size_);
    pw_ndt.setIntensityBias(bias_);
    pw_ndt.setInputSource (old_in_pc);
    pw_ndt.setInputTarget (in_pc);
    pw_ndt.align (*output_cloud, T);
    score = pw_ndt.getFitnessScore ();
    T = pw_ndt.getFinalTransformation();
    normal_distributions_map = pw_ndt.getNormalDistributionMap();
  }
  else{
    // Initializing Normal Distributions Transform (NDT).
    pcl::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI> ndt;
    ndt.setTransformationEpsilon (0.01);
    ndt.setStepSize (1);
    ndt.setResolution (10);
    ndt.setMaximumIterations (300);
    pcl::PointCloud<pcl::PointXYZI>::Ptr old_in_pc_flat (new pcl::PointCloud<pcl::PointXYZI>);
    old_in_pc_flat = make2DPointCloud(old_in_pc);
    pcl::PointCloud<pcl::PointXYZI>::Ptr in_pc_flat (new pcl::PointCloud<pcl::PointXYZI>);
    in_pc_flat = make2DPointCloud(in_pc);
    ndt.setInputSource (old_in_pc_flat);
    ndt.setInputTarget (in_pc_flat);
    ndt.align (*output_cloud);
    std::cout << "Normal Distributions Transform has converged:" << ndt.hasConverged ()
                  << " score: " << ndt.getFitnessScore () << std::endl;
    T = ndt.getFinalTransformation ();
  }
}

void deleteNDMap(){
  MarkerArray ret;
  Marker m;
  m.action = Marker::DELETEALL;
  ret.markers.push_back(m);
  nd_map_pub.publish(ret);
}

void pubNDMap(Eigen::Matrix< pcl::ndt2d::NormalDist<pcl::PointXYZI> , Eigen::Dynamic, Eigen::Dynamic> normal_distributions_map,
              double scale){
  vector<Marker> marker_vec;
  for (size_t i=0; i<normal_distributions_map.rows(); i++){
    for (size_t j=0; j<normal_distributions_map.cols(); j++){
      Eigen::Vector2d u = normal_distributions_map(i,j).getMean();
      Eigen::Matrix2d cov = normal_distributions_map(i,j).getCovar();
      if(u[0] != 0 && u[1] != 0){
        Marker marker;
        marker = MarkerOfEllipse(u, cov, Color::kAqua, 0.25, scale*2);
        marker_vec.push_back(marker);
        marker = MarkerOfEllipse(u, cov, Color::kAqua, 0.25, scale*3);
        marker_vec.push_back(marker);
        marker = MarkerOfEllipse(u, cov, Color::kAqua, 0.25, scale*4);
        marker_vec.push_back(marker);

//          kRed,    /**< (R, G, B) = (1.0, 0.0, 0.0) */
//          kLime,   /**< (R, G, B) = (0.0, 1.0, 0.0) */
//          kBlue,   /**< (R, G, B) = (0.0, 0.0, 1.0) */
//          kWhite,  /**< (R, G, B, A) = (x, x, x, 0.0) */
//          kBlack,  /**< (R, G, B) = (0.0, 0.0, 0.0) */
//          kGray,   /**< (R, G, B) = (0.5, 0.5, 0.5) */
//          kYellow, /**< (R, G, B) = (1.0, 1.0, 0.0) */
//          kAqua,   /**< (R, G, B) = (0.0, 1.0, 1.0) */
//          kFuchsia /**< (R, G, B) = (1.0, 0.0, 1.0) */

      }
    }
  }
  MarkerArray marker_array;
  marker_array = JoinMarkers(marker_vec);
  nd_map_pub.publish(marker_array);
}


bool CheckAcc(Eigen::Matrix4f& T){
  float curr_vel = VelocityCal(T, curr_t - old_t);
  float acc = (curr_vel-old_vel)/(curr_t-old_t);

  float curr_ang_vel = AngularVelocityCal(T, curr_t - old_t);
  float ang_acc = (curr_ang_vel-old_ang_vel)/(curr_t-old_t);

  if(curr_vel != curr_vel || curr_ang_vel != curr_ang_vel){
    return 1;
  }

  if(abs(acc) > 8 || abs(ang_acc) > 50){
      ROS_WARN("[%d] abnormal acc: %f", Index, acc);
      return 1;
  }
  else{
    return 0;
  }
}

void PrintAcc(Eigen::Matrix4f& T){
  float x_gt = T(0,3); float y_gt = T(1,3); float theta_gt = atan2(T(1,0), T(0,0));
  cout << " xytheta: " << x_gt << " " << y_gt << " " << theta_gt*180/3.14 << endl;
  float curr_vel = VelocityCal(T, curr_t - old_t);
  float acc = (curr_vel-old_vel)/(curr_t-old_t);
  float curr_ang_vel = AngularVelocityCal(T, curr_t - old_t);
  float ang_acc = (curr_ang_vel-old_ang_vel)/(curr_t-old_t);
  cout << " vel: " << curr_vel << " ang_vel: " << curr_ang_vel << " ";
  cout << " acc: " << acc << " ang_acc: " << ang_acc << endl;
}

///--- Visualization ---///
void Vizsualization(bool viz,
                    const pcl::PointCloud<pcl::PointXYZI>::Ptr in_pc,
                    const pcl::PointCloud<pcl::PointXYZI>::Ptr output_cloud,
                    const pcl::PointCloud<pcl::PointXYZI>::Ptr old_in_pc){
  if(viz){
    // Initializing point cloud visualizer
    pcl::visualization::PCLVisualizer::Ptr
    viewer_final (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer_final->setBackgroundColor (0.1, 0.1, 0.1);

    // Coloring and visualizing target cloud (red).
    pcl::PointCloud<pcl::PointXYZI>::Ptr in_pc_flat (new pcl::PointCloud<pcl::PointXYZI>);
    in_pc_flat = make2DPointCloud(in_pc);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI>
    target_color (in_pc_flat, 255, 0, 0);
    viewer_final->addPointCloud<pcl::PointXYZI> (in_pc_flat, target_color, "target_cloud");

    // Coloring and visualizing transformed input cloud (green).
    pcl::PointCloud<pcl::PointXYZI>::Ptr output_cloud_flat (new pcl::PointCloud<pcl::PointXYZI>);
    output_cloud_flat = make2DPointCloud(output_cloud);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI>
    output_color (output_cloud_flat, 0, 255, 0);
    viewer_final->addPointCloud<pcl::PointXYZI> (output_cloud_flat, output_color, "output_cloud");

    // Coloring and visualizing transformed input cloud (blue).
    pcl::PointCloud<pcl::PointXYZI>::Ptr old_in_pc_flat (new pcl::PointCloud<pcl::PointXYZI>);
    old_in_pc_flat = make2DPointCloud(old_in_pc);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI>
    input_color (old_in_pc_flat, 0, 30, 255);
    viewer_final->addPointCloud<pcl::PointXYZI> (old_in_pc_flat, input_color, "input_cloud");

    // Starting visualizer
    viewer_final->addCoordinateSystem (1.0, "global");
    viewer_final->initCameraParameters ();
    viewer_final->setCameraPosition (0,0,400,0,0,0,0,0,0);

    // Wait until visualizer window is closed.
    while (!viewer_final->wasStopped ())
    {
      viewer_final->spinOnce (100);
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  }

}
