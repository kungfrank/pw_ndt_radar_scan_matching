#include <iostream>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <limits>

#include "mail.h"
#include "matrix.h"

using namespace std;

float lengths[] = {100,200,300,400,500,600,700,800};
int32_t num_lengths = 8;

string bag_name[] = {"2019-01-10-11-46-21-radar-oxford-10k",
                     "2019-01-10-12-32-52-radar-oxford-10k", "2019-01-10-14-02-34-radar-oxford-10k", "2019-01-10-14-50-05-radar-oxford-10k",
                     "2019-01-10-15-19-41-radar-oxford-10k", "2019-01-11-12-26-55-radar-oxford-10k", "2019-01-11-13-24-51-radar-oxford-10k",
                     "2019-01-11-14-02-26-radar-oxford-10k", "2019-01-14-12-05-52-radar-oxford-10k", "2019-01-14-13-38-21-radar-oxford-10k",
                     "2019-01-14-14-15-12-radar-oxford-10k", "2019-01-14-14-48-55-radar-oxford-10k", "2019-01-15-12-01-32-radar-oxford-10k",
                     "2019-01-15-13-06-37-radar-oxford-10k", "2019-01-15-13-53-14-radar-oxford-10k", "2019-01-15-14-24-38-radar-oxford-10k",
                     "2019-01-16-11-53-11-radar-oxford-10k", "2019-01-16-13-09-37-radar-oxford-10k", "2019-01-16-13-42-28-radar-oxford-10k",
                     "2019-01-16-14-15-33-radar-oxford-10k", "2019-01-17-11-46-31-radar-oxford-10k", "2019-01-17-12-48-25-radar-oxford-10k",
                     "2019-01-17-13-26-39-radar-oxford-10k", "2019-01-17-14-03-00-radar-oxford-10k", "2019-01-18-12-42-34-radar-oxford-10k",
                     "2019-01-18-14-14-42-radar-oxford-10k", "2019-01-18-14-46-59-radar-oxford-10k", "2019-01-18-15-20-12-radar-oxford-10k"};

struct errors {
  int32_t first_frame;
  float   r_err;
  float   t_err;
  float   len;
  float   speed;
  errors (int32_t first_frame,float r_err,float t_err,float len,float speed) :
    first_frame(first_frame),r_err(r_err),t_err(t_err),len(len),speed(speed) {}
};

vector<Matrix> loadPoses(string file_name) {
  cout << "loadPoses: " << file_name << endl;
  vector<Matrix> poses;
  FILE *fp = fopen(file_name.c_str(),"r");
  if (!fp)
    return poses;
  while (!feof(fp)) {
    Matrix P = Matrix::eye(4);
    if (fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                   &P.val[0][0], &P.val[0][1], &P.val[0][2], &P.val[0][3],
                   &P.val[1][0], &P.val[1][1], &P.val[1][2], &P.val[1][3],
                   &P.val[2][0], &P.val[2][1], &P.val[2][2], &P.val[2][3] )==12) {
      poses.push_back(P);
    }
  }
  fclose(fp);
  ///////////////////////// REMOVE LAST TERM /////////////////////////
  poses.pop_back();
  ///////////////////////// REMOVE LAST TERM /////////////////////////
  return poses;
}

vector<Matrix> loadGtPoses(string file_name) {
  cout << "loadGtPoses: " << file_name << endl;
  vector<Matrix> poses;
  FILE *fp = fopen((file_name).c_str(),"r");
  
  if (!fp)
    return poses;
  while (!feof(fp)) {
    Matrix P = Matrix::eye(4);
    if (fscanf(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                   &P.val[0][0], &P.val[0][1], &P.val[0][2], &P.val[0][3],
                   &P.val[1][0], &P.val[1][1], &P.val[1][2], &P.val[1][3],
                   &P.val[2][0], &P.val[2][1], &P.val[2][2], &P.val[2][3] )==12) {
      poses.push_back(P);
    }
  }
  fclose(fp);
  return poses;
}

vector<float> trajectoryDistances (vector<Matrix> &poses) {
  vector<float> dist;
  dist.push_back(0);
  for (int32_t i=1; i<poses.size(); i++) {
    Matrix P1 = poses[i-1];
    Matrix P2 = poses[i];
    float dx = P1.val[0][3]-P2.val[0][3];
    float dy = P1.val[1][3]-P2.val[1][3];
    float dz = P1.val[2][3]-P2.val[2][3];
    dist.push_back(dist[i-1]+sqrt(dx*dx+dy*dy+dz*dz));
  }
  return dist;
}

int32_t lastFrameFromSegmentLength(vector<float> &dist,int32_t first_frame,float len) {
  for (int32_t i=first_frame; i<dist.size(); i++)
    if (dist[i]>dist[first_frame]+len)
      return i;
  return -1;
}

inline float rotationError(Matrix &pose_error) {
  float a = pose_error.val[0][0];
  float b = pose_error.val[1][1];
  float c = pose_error.val[2][2];
  float d = 0.5*(a+b+c-1.0);
  return acos(max(min(d,1.0f),-1.0f));
}

inline float translationError(Matrix &pose_error) {
  float dx = pose_error.val[0][3];
  float dy = pose_error.val[1][3];
  float dz = pose_error.val[2][3];
  return sqrt(dx*dx+dy*dy+dz*dz);
}

vector<errors> calcSequenceErrors (vector<Matrix> &poses_gt,vector<Matrix> &poses_result) {

  // error vector
  vector<errors> err;

  // parameters
  int32_t step_size = 4; // every second
  //int32_t step_size = 1; // every frame

  // pre-compute distances (from ground truth as reference)
  vector<float> dist = trajectoryDistances(poses_gt);

  // for all start positions do
  for (int32_t first_frame=0; first_frame<poses_gt.size(); first_frame+=step_size) {
  
    // for all segment lengths do
    for (int32_t i=0; i<num_lengths; i++) {
    
      // current length
      float len = lengths[i];
      // compute last frame
      int32_t last_frame = lastFrameFromSegmentLength(dist,first_frame,len);
      // continue, if sequence not long enough
      if (last_frame==-1)
        continue;

      // compute rotational and translational errors
      Matrix pose_delta_gt     = Matrix::inv(poses_gt[first_frame])*poses_gt[last_frame];
      Matrix pose_delta_result = Matrix::inv(poses_result[first_frame])*poses_result[last_frame];
      Matrix pose_error        = Matrix::inv(pose_delta_result)*pose_delta_gt;
      float r_err = rotationError(pose_error);
      float t_err = translationError(pose_error);

      // -- FOR DEBUG -- //
      /*
			if(i==1 && first_frame ==0){
				std::cout << "pose_delta_gt: \n" << pose_delta_gt << std::endl;
				std::cout << "pose_delta_result: \n" << pose_delta_result << std::endl;
				std::cout << "pose_error: \n" << pose_error << std::endl;
			}
      */
      // compute speed
      float num_frames = (float)(last_frame-first_frame+1);
      float speed = len/(0.1*num_frames);
      // write to file
      err.push_back(errors(first_frame,r_err/len,t_err/len,len,speed));
    }
  }
  // return error vector
  return err;
}

void saveSequenceErrors (vector<errors> &err,string file_name) {

  // open file  
  FILE *fp;
  fp = fopen(file_name.c_str(),"w");

  // write to file
  for (vector<errors>::iterator it=err.begin(); it!=err.end(); it++)
    fprintf(fp,"%d %f %f %f %f\n",it->first_frame,it->r_err,it->t_err,it->len,it->speed);
  
  // close file
  fclose(fp);
}

void savePathPlot (vector<Matrix> &poses_gt,vector<Matrix> &poses_result,string file_name) {

  // parameters
  int32_t step_size = 3;

  // open file  
  FILE *fp = fopen(file_name.c_str(),"w");
 
  // save x/y coordinates of all frames to file
  for (int32_t i=0; i<poses_gt.size(); i+=step_size)
    fprintf(fp,"%f %f %f %f\n",poses_gt[i].val[0][3],poses_gt[i].val[1][3],
                               poses_result[i].val[0][3],poses_result[i].val[1][3]);
  
  // close file
  fclose(fp);
}

vector<int32_t> computeRoi (vector<Matrix> &poses_gt,vector<Matrix> &poses_result) {
  
  float x_min = numeric_limits<int32_t>::max();
  float x_max = numeric_limits<int32_t>::min();
  float z_min = numeric_limits<int32_t>::max(); // This is y !!!
  float z_max = numeric_limits<int32_t>::min(); // This is y !!!
  
  for (vector<Matrix>::iterator it=poses_gt.begin(); it!=poses_gt.end(); it++) {
    float x = it->val[0][3];
    float z = it->val[1][3];
    if (x<x_min) x_min = x; if (x>x_max) x_max = x;
    if (z<z_min) z_min = z; if (z>z_max) z_max = z;
  }
  
  for (vector<Matrix>::iterator it=poses_result.begin(); it!=poses_result.end(); it++) {
    float x = it->val[0][3];
    float z = it->val[1][3];
    if (x<x_min) x_min = x; if (x>x_max) x_max = x;
    if (z<z_min) z_min = z; if (z>z_max) z_max = z;
  }
  
  float dx = 1.1*(x_max-x_min);
  float dz = 1.1*(z_max-z_min);
  float mx = 0.5*(x_max+x_min);
  float mz = 0.5*(z_max+z_min);
  float r  = 0.5*max(dx,dz);
  
  vector<int32_t> roi;
  roi.push_back((int32_t)(mx-r));
  roi.push_back((int32_t)(mx+r));
  roi.push_back((int32_t)(mz-r));
  roi.push_back((int32_t)(mz+r));
  return roi;
}

void plotPathPlot (string dir,vector<int32_t> &roi,int32_t idx) {

  // gnuplot file name
  char command[1024];
  char file_name[256];
  sprintf(file_name,"%02d.gp",idx);
  string full_name = dir + "/" + file_name;
  
  // create png + eps
  for (int32_t i=0; i<2; i++) {

    // open file  
    FILE *fp = fopen(full_name.c_str(),"w");

    // save gnuplot instructions
    if (i==0) {
      fprintf(fp,"set term png size 900,900\n");
      fprintf(fp,"set output \"%02d.png\"\n",idx);
    } else {
      fprintf(fp,"set term postscript eps enhanced color\n");
      fprintf(fp,"set output \"%02d.eps\"\n",idx);
    }

    fprintf(fp,"set size ratio -1\n");
    fprintf(fp,"set xrange [%d:%d]\n",roi[0],roi[1]);
    fprintf(fp,"set yrange [%d:%d]\n",roi[2],roi[3]);
    fprintf(fp,"set xlabel \"x [m]\"\n");
    fprintf(fp,"set ylabel \"y [m]\"\n");  // y
    fprintf(fp,"plot \"%02d.txt\" using 1:2 lc rgb \"#FF0000\" title 'Ground Truth' w lines,",idx);
    fprintf(fp,"\"%02d.txt\" using 3:4 lc rgb \"#0000FF\" title 'Radar Odometry' w lines,",idx);
    fprintf(fp,"\"< head -1 %02d.txt\" using 1:2 lc rgb \"#000000\" pt 4 ps 1 lw 2 title 'Sequence Start' w points\n",idx);
    
    // close file
    fclose(fp);

    // run gnuplot => create png + eps
    sprintf(command,"cd %s; gnuplot %s",dir.c_str(),file_name);
    system(command);
  }
  
  // create pdf and crop
  sprintf(command,"cd %s; ps2pdf %02d.eps %02d_large.pdf",dir.c_str(),idx,idx);
  system(command);
  sprintf(command,"cd %s; pdfcrop %02d_large.pdf %02d.pdf",dir.c_str(),idx,idx);
  system(command);
  sprintf(command,"cd %s; rm %02d_large.pdf",dir.c_str(),idx);
  system(command);
}

void saveErrorPlots(vector<errors> &seq_err,string plot_error_dir,char* prefix) {

  // file names
  char file_name_tl[1024]; sprintf(file_name_tl,"%s/%s_tl.txt",plot_error_dir.c_str(),prefix);
  char file_name_rl[1024]; sprintf(file_name_rl,"%s/%s_rl.txt",plot_error_dir.c_str(),prefix);
  char file_name_ts[1024]; sprintf(file_name_ts,"%s/%s_ts.txt",plot_error_dir.c_str(),prefix);
  char file_name_rs[1024]; sprintf(file_name_rs,"%s/%s_rs.txt",plot_error_dir.c_str(),prefix);

  // open files
  FILE *fp_tl = fopen(file_name_tl,"w");
  FILE *fp_rl = fopen(file_name_rl,"w");
  FILE *fp_ts = fopen(file_name_ts,"w");
  FILE *fp_rs = fopen(file_name_rs,"w");
 
  // for each segment length do
  for (int32_t i=0; i<num_lengths; i++) {

    float t_err = 0;
    float r_err = 0;
    float num   = 0;

    // for all errors do
    for (vector<errors>::iterator it=seq_err.begin(); it!=seq_err.end(); it++) {
      if (fabs(it->len-lengths[i])<1.0) {
        t_err += it->t_err;
        r_err += it->r_err;
        num++;
      }
    }
    
    // we require at least 3 values
    if (num>2.5) {
      fprintf(fp_tl,"%f %f\n",lengths[i],t_err/num);
      fprintf(fp_rl,"%f %f\n",lengths[i],r_err/num);
    }
  }
  
  // for each driving speed do (in m/s)
  for (float speed=2; speed<25; speed+=2) {

    float t_err = 0;
    float r_err = 0;
    float num   = 0;

    // for all errors do
    for (vector<errors>::iterator it=seq_err.begin(); it!=seq_err.end(); it++) {
      if (fabs(it->speed-speed)<2.0) {
        t_err += it->t_err;
        r_err += it->r_err;
        num++;
      }
    }
    
    // we require at least 3 values
    if (num>2.5) {
      fprintf(fp_ts,"%f %f\n",speed,t_err/num);
      fprintf(fp_rs,"%f %f\n",speed,r_err/num);
    }
  }
  
  // close files
  fclose(fp_tl);
  fclose(fp_rl);
  fclose(fp_ts);
  fclose(fp_rs);
}

void saveIndexErrorPlots(vector<errors> &seq_err,string plot_error_dir,char* prefix) {

  // file names
  char file_name_ti[1024]; sprintf(file_name_ti,"%s/%s_ti.txt",plot_error_dir.c_str(),prefix);
  char file_name_ri[1024]; sprintf(file_name_ri,"%s/%s_ri.txt",plot_error_dir.c_str(),prefix);

  // open files
  FILE *fp_ti = fopen(file_name_ti,"w");
  FILE *fp_ri = fopen(file_name_ri,"w");

  int32_t last_index = 0;
  float t_err = 0;
  float r_err = 0;
  float num   = 0;

  for (vector<errors>::iterator it=seq_err.begin(); it!=seq_err.end(); it++){
      int32_t index = it->first_frame;
      if(index == last_index){
          t_err += it->t_err;
          r_err += it->r_err;
          num++;
      }
      else{
          fprintf(fp_ti,"%d %f\n",last_index,t_err/num);
          fprintf(fp_ri,"%d %f\n",last_index,r_err/num);

          t_err = 0;
          r_err = 0;
          num   = 0;
          t_err += it->t_err;
          r_err += it->r_err;
          num++;
      }
      last_index = index;
  }

  // close files
  fclose(fp_ti);
  fclose(fp_ri);
}

void plotErrorPlots (string dir,char* prefix) {

  char command[1024];

  // for all four error plots do
  for (int32_t i=0; i<4; i++) {
 
    // create suffix
    char suffix[16];
    switch (i) {
      case 0: sprintf(suffix,"tl"); break;
      case 1: sprintf(suffix,"rl"); break;
      case 2: sprintf(suffix,"ts"); break;
      case 3: sprintf(suffix,"rs"); break;
    }
       
    // gnuplot file name
    char file_name[1024]; char full_name[1024];
    sprintf(file_name,"%s_%s.gp",prefix,suffix);
    sprintf(full_name,"%s/%s",dir.c_str(),file_name);
    
    // create png + eps
    for (int32_t j=0; j<2; j++) {

      // open file  
      FILE *fp = fopen(full_name,"w");

      // save gnuplot instructions
      if (j==0) {
        fprintf(fp,"set term png size 500,250 font \"Helvetica\" 11\n");
        fprintf(fp,"set output \"%s_%s.png\"\n",prefix,suffix);
      } else {
        fprintf(fp,"set term postscript eps enhanced color\n");
        fprintf(fp,"set output \"%s_%s.eps\"\n",prefix,suffix);
      }
      
      // start plot at 0
      fprintf(fp,"set size ratio 0.5\n");
      fprintf(fp,"set yrange [0:*]\n");

      // x label
      if (i<=1) fprintf(fp,"set xlabel \"Path Length [m]\"\n");
      else      fprintf(fp,"set xlabel \"Speed [km/h]\"\n");
      
      // y label
      if (i==0 || i==2) fprintf(fp,"set ylabel \"Translation Error [%%]\"\n");
      else              fprintf(fp,"set ylabel \"Rotation Error [deg/m]\"\n");
      
      // plot error curve
      fprintf(fp,"plot \"%s_%s.txt\" using ",prefix,suffix);
      switch (i) {
        case 0: fprintf(fp,"1:($2*100) title 'Translation Error'"); break;
        case 1: fprintf(fp,"1:($2*57.3) title 'Rotation Error'"); break;
        case 2: fprintf(fp,"($1*3.6):($2*100) title 'Translation Error'"); break;
        case 3: fprintf(fp,"($1*3.6):($2*57.3) title 'Rotation Error'"); break;
      }
      fprintf(fp," lc rgb \"#0000FF\" pt 4 w linespoints\n");
      
      // close file
      fclose(fp);
      
      // run gnuplot => create png + eps
      sprintf(command,"cd %s; gnuplot %s",dir.c_str(),file_name);
      system(command);
    }
    
    // create pdf and crop
    sprintf(command,"cd %s; ps2pdf %s_%s.eps %s_%s_large.pdf",dir.c_str(),prefix,suffix,prefix,suffix);
    system(command);
    sprintf(command,"cd %s; pdfcrop %s_%s_large.pdf %s_%s.pdf",dir.c_str(),prefix,suffix,prefix,suffix);
    system(command);
    sprintf(command,"cd %s; rm %s_%s_large.pdf",dir.c_str(),prefix,suffix);
    system(command);
  }
}

void plotIndexErrorPlots (string dir,char* prefix) {

  char command[1024];

  // for all four error plots do
  for (int32_t i=0; i<2; i++) {

    // create suffix
    char suffix[16];
    switch (i) {
      case 0: sprintf(suffix,"ti"); break;
      case 1: sprintf(suffix,"ri"); break;
    }

    // gnuplot file name
    char file_name[1024]; char full_name[1024];
    sprintf(file_name,"%s_%s.gp",prefix,suffix);
    sprintf(full_name,"%s/%s",dir.c_str(),file_name);

    // create png + eps
    for (int32_t j=0; j<2; j++) {

      // open file
      FILE *fp = fopen(full_name,"w");

      // save gnuplot instructions
      if (j==0) {
        fprintf(fp,"set term png size 500,250 font \"Helvetica\" 11\n");
        fprintf(fp,"set output \"%s_%s.png\"\n",prefix,suffix);
      } else {
        fprintf(fp,"set term postscript eps enhanced color\n");
        fprintf(fp,"set output \"%s_%s.eps\"\n",prefix,suffix);
      }

      // start plot at 0
      fprintf(fp,"set size ratio 1\n");
      fprintf(fp,"set yrange [0:*]\n");

      // x label
      fprintf(fp,"set xlabel \"Index\"\n");

      // y label
      if (i==0) fprintf(fp,"set ylabel \"Translation Error [%%]\"\n");
      else      fprintf(fp,"set ylabel \"Rotation Error [deg/m]\"\n");

      // plot error curve
      fprintf(fp,"plot \"%s_%s.txt\" using ",prefix,suffix);
      switch (i) {
        case 0: fprintf(fp,"1:($2*100) title 'Translation Error'"); break;
        case 1: fprintf(fp,"1:($2*57.3) title 'Rotation Error'"); break;
        //case 2: fprintf(fp,"($1*3.6):($2*100) title 'Translation Error'"); break;
        //case 3: fprintf(fp,"($1*3.6):($2*57.3) title 'Rotation Error'"); break;
      }
      fprintf(fp," lc rgb \"#0000FF\" pt 2 w lines\n");

      // close file
      fclose(fp);

      // run gnuplot => create png + eps
      sprintf(command,"cd %s; gnuplot %s",dir.c_str(),file_name);
      system(command);
    }

    // create pdf and crop
    sprintf(command,"cd %s; ps2pdf %s_%s.eps %s_%s_large.pdf",dir.c_str(),prefix,suffix,prefix,suffix);
    system(command);
    sprintf(command,"cd %s; pdfcrop %s_%s_large.pdf %s_%s.pdf",dir.c_str(),prefix,suffix,prefix,suffix);
    system(command);
    sprintf(command,"cd %s; rm %s_%s_large.pdf",dir.c_str(),prefix,suffix);
    system(command);

  }
}

void saveStats (vector<errors> err,string dir) {

  float t_err = 0;
  float r_err = 0;

  // for all errors do => compute sum of t_err, r_err
  for (vector<errors>::iterator it=err.begin(); it!=err.end(); it++) {
    t_err += it->t_err;
    r_err += it->r_err;
  }

  // open file  
  FILE *fp = fopen((dir + "/stats.txt").c_str(),"a");
 
  // save errors
  float num = err.size();
  fprintf(fp,"  %f %f\n",t_err/num,r_err/num);
  
  // close file
  fclose(fp);
}

void printStats (vector<errors> err, string dir) {

  float t_err = 0;
  float r_err = 0;

  // for all errors do => compute sum of t_err, r_err
  for (vector<errors>::iterator it=err.begin(); it!=err.end(); it++) {
    t_err += it->t_err;
    r_err += it->r_err;
  }

  // open file
  FILE *fp = fopen((dir + "/stats.txt").c_str(),"a");

  // save errors
  float num = err.size();
  fprintf(fp,"%f %f\n",t_err/num,r_err/num);
  printf("Stats: %f %f \n \n",t_err/num,r_err/num);

  // close file
  fclose(fp);
}

bool eval (string result_sha,Mail* mail) {

  // ground truth and result directories
  string gt_dir         = "data/odometry/poses";
  string result_dir     = "results/" + result_sha;
  string error_dir      = result_dir + "/errors";
  string plot_path_dir  = result_dir + "/plot_path";
  string plot_error_dir = result_dir + "/plot_error";

  // create output directories
  system(("mkdir " + error_dir).c_str());
  system(("mkdir " + plot_path_dir).c_str());
  system(("mkdir " + plot_error_dir).c_str());
  
  // total errors
  vector<errors> total_err;

  // for all sequences do
  for (int32_t i=0; i<28; i++) { //28

    // file name
    string file_name;
    file_name = bag_name[i];

    // read ground truth and result poses
    //string file_name_first_part = file_name.substr(0, 16);

    vector<Matrix> poses_gt     = loadGtPoses(gt_dir + "/" + file_name + "_gt.txt");
    vector<Matrix> poses_result = loadPoses(result_dir + "/data/" + file_name + "_res.txt");

    // plot status
    mail->msg("Processing: %s, poses: %d/%d",file_name.c_str(),poses_result.size(),poses_gt.size());
    
    // check for errors
    if (poses_gt.size()==0 || poses_result.size()!=poses_gt.size()) {
      mail->msg("ERROR: Couldn't read (all) poses of: %s", file_name);
      return false;
    }

    // compute sequence errors    
    vector<errors> seq_err = calcSequenceErrors(poses_gt,poses_result);

    saveSequenceErrors(seq_err,error_dir + "/" + file_name);

    //--- ADD BY FRANK ---//
    printStats(seq_err, result_dir);
    
    // add to total errors
    total_err.insert(total_err.end(),seq_err.begin(),seq_err.end());
    
    // for first half => plot trajectory and compute individual stats
    if (i<=15) {
    
      // save + plot bird's eye view trajectories
      char plot_file_name[256];
      sprintf(plot_file_name,"%02d.txt",i);

      savePathPlot(poses_gt,poses_result,plot_path_dir + "/" + plot_file_name);
      vector<int32_t> roi = computeRoi(poses_gt,poses_result);
      plotPathPlot(plot_path_dir,roi,i);

      // save + plot individual errors
      char prefix[16];
      sprintf(prefix,"%02d",i);
      saveErrorPlots(seq_err,plot_error_dir,prefix);
      plotErrorPlots(plot_error_dir,prefix);

      //--- ADD BY FRANK ---//
      saveIndexErrorPlots(seq_err,plot_error_dir,prefix);
      plotIndexErrorPlots(plot_error_dir,prefix);
    }
  }
  
  // save + plot total errors + summary statistics
  if (total_err.size()>0) {
    char prefix[16];
    sprintf(prefix,"avg");
    saveErrorPlots(total_err,plot_error_dir,prefix);
    plotErrorPlots(plot_error_dir,prefix);
    saveStats(total_err,result_dir);
  }

  // success
	return true;
}

int32_t main (int32_t argc,char *argv[]) {

  // we need 2 or 4 arguments!
  if (argc!=2 && argc!=4) {
    cout << "Usage: ./eval_odometry result_sha [user_sha email]" << endl;
    return 1;
  }

  // read arguments
  string result_sha = argv[1];

  // init notification mail
  Mail *mail;
  if (argc==4) mail = new Mail(argv[3]);
  else         mail = new Mail();
  mail->msg("Thank you for participating in our evaluation!");

  // run evaluation
  bool success = eval(result_sha,mail);
  if (argc==4) mail->finalize(success,"odometry",result_sha,argv[2]);
  else         mail->finalize(success,"odometry",result_sha);

  // send mail and exit
  delete mail;
  return 0;
}

