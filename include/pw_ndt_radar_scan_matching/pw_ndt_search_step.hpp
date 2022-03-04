/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2011-2012, Willow Garage, Inc.
 *  Copyright (c) 2012-, Open Perception, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id$
 *
 */
#ifndef PCL_NDT_2D_IMPL_H_
#define PCL_NDT_2D_IMPL_H_
#include <cmath>

#include <pcl/registration/eigen.h>
#include <pcl/registration/boost.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <chrono>

namespace Eigen
{
  /* This NumTraits specialisation is necessary because NormalDist is used as
   * the element type of an Eigen Matrix.
   */
  template<typename PointT> struct NumTraits<pcl::ndt2d::NormalDist<PointT> >
  {
    typedef double Real;
    static Real dummy_precision () { return 1.0; }
    enum {
      IsComplex = 0,
      IsInteger = 0,
      IsSigned = 0,
      RequireInitialization = 1,
      ReadCost = 1,
      AddCost = 1,
      MulCost = 1
    };
  };
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename PointSource, typename PointTarget> void
pcl::PWNormalDistributionsTransform2D<PointSource, PointTarget>::computeTransformation (PointCloudSource &output, const Eigen::Matrix4f &guess)
{
  PointCloudSource intm_cloud = output;

  if (guess != Eigen::Matrix4f::Identity ())
  {
    transformation_ = guess;
    transformPointCloud (output, intm_cloud, transformation_);
  }

  // build Normal Distribution Transform of target cloud:
  ndt2d::NDT2D<PointTarget> target_ndt (target_, grid_centre_, grid_extent_, grid_step_, bias_, gamma_, normal_distributions_map_);
  
  // can't seem to use .block<> () member function on transformation_
  // directly... gcc bug? 
  Eigen::Matrix4f& transformation = transformation_;

  // work with x translation, y translation and z rotation: extending to 3D
  // would be some tricky maths, but not impossible.
  const Eigen::Matrix3f initial_rot (transformation.block<3,3> (0,0));
  const Eigen::Vector3f rot_x (initial_rot*Eigen::Vector3f::UnitX ());
  const double z_rotation = std::atan2 (rot_x[1], rot_x[0]);

  Eigen::Vector3d xytheta_transformation (
    transformation (0,3),
    transformation (1,3),
    z_rotation
  );

  ndt2d::ValueAndDerivatives<3, double>
  score = computeDerivatives(target_ndt, intm_cloud, xytheta_transformation);

  while (!converged_)
  {
    previous_transformation_ = transformation;

//    std::cout << " [NDT] ";
//    std::cout << "------------------------------- Start iterations_: " << nr_iterations_  << " ------------------------------- \n";
//    std::cout << "score.value: " << score.value  << "\n" <<
//                 "score.grad: \n" << score.grad.transpose()  << "\n";

    if (score.value != 0){

      // test for positive definiteness, and adjust to ensure it if necessary:
      Eigen::EigenSolver<Eigen::Matrix3d> solver;
      solver.compute (score.hessian, false);
      double min_eigenvalue = 0;
      for (int i = 0; i <3; i++)
        if (solver.eigenvalues ()[i].real () < min_eigenvalue)
          min_eigenvalue = solver.eigenvalues ()[i].real ();

      ////---- Check Hessian Matrix ----////
      // ensure "safe" positive definiteness: this is a detail missing
      // from the original paper
      if (min_eigenvalue < 0)
      {
        double lambda = 1.1 * min_eigenvalue - 1;
        score.hessian += Eigen::Vector3d (-lambda, -lambda, -lambda).asDiagonal ();
        solver.compute (score.hessian, false);
        PCL_DEBUG ("[pcl::NormalDistributionsTransform2D::computeTransformation] adjust hessian: %f: new eigenvalues:%f %f %f\n",
            float (lambda),
            solver.eigenvalues ()[0].real (),
            solver.eigenvalues ()[1].real (),
            solver.eigenvalues ()[2].real ()
        );
      }
      assert (solver.eigenvalues ()[0].real () >= 0 &&
              solver.eigenvalues ()[1].real () >= 0 &&
              solver.eigenvalues ()[2].real () >= 0);

      ////---- Check Condition number of Hessian Matrix ----////
      double min_hessian_eigvalue_mult = 0.00000001;
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver_for_hessian (score.hessian);
      //cout << "solver_for_hessian.eigenvalues (): " << solver_for_hessian.eigenvalues ().transpose() << endl;
      //cout << "             eigenvalue_max/eigenvalue_min: " << solver_for_hessian.eigenvalues ()[2] / solver_for_hessian.eigenvalues ()[0] << endl;

      if (solver_for_hessian.eigenvalues ()[0] < min_hessian_eigvalue_mult * solver_for_hessian.eigenvalues ()[2])
      {
        PCL_DEBUG ("[pcl::NormalDist::estimateParams] NDT normal fit: adjusting eigenvalue %f\n", solver_for_hessian.eigenvalues ()[0]);
        Eigen::Matrix3d l = solver_for_hessian.eigenvalues ().asDiagonal ();
        Eigen::Matrix3d q = solver_for_hessian.eigenvectors ();
        // set minimum smallest eigenvalue:
        l (0,0) = l (2,2) * min_hessian_eigvalue_mult;
        score.hessian = q * l * q.transpose ();
      }

      ////---- Compute delta_transformation ----////
      Eigen::Vector3d delta_transformation (-score.hessian.inverse () * score.grad);
      //cout << "delta_transformation: " << delta_transformation.transpose() << " delta_transformation.norm (): " << delta_transformation.norm () << endl;

      ////---- Newton Method ----////
//      Eigen::Vector3d new_transformation = xytheta_transformation + newton_lambda_.cwiseProduct (delta_transformation);
//      xytheta_transformation = new_transformation;

      ////---- Search Step [More, Thuente 1994] ----////
      // Calculate step length with guarnteed sufficient decrease
      double delta_transformation_norm;
      delta_transformation_norm = delta_transformation.norm ();
      if (delta_transformation_norm == 0 || delta_transformation_norm != delta_transformation_norm)
      {
        converged_ = delta_transformation_norm == delta_transformation_norm;
        return;
      }
      delta_transformation.normalize ();
      delta_transformation_norm = computeStepLengthMT (xytheta_transformation, delta_transformation, delta_transformation_norm /*Init Step*/, max_step_size_ /*Max Step*/, transformation_epsilon_/2 /*Min Step*/,
                                                       score, output, intm_cloud, target_ndt);
      delta_transformation *= delta_transformation_norm;
      xytheta_transformation = xytheta_transformation + delta_transformation;
      //cout << "xytheta_transformation: " << xytheta_transformation.transpose() << endl;


      transformation.block<3,3> (0,0).matrix () = Eigen::Matrix3f (Eigen::AngleAxisf (static_cast<float> (xytheta_transformation(2)), Eigen::Vector3f::UnitZ ()));
      transformation.block<3,1> (0,3).matrix () = Eigen::Vector3f (static_cast<float> (xytheta_transformation(0)), static_cast<float> (xytheta_transformation(1)), 0.0f);
    }
    else
    {
      PCL_ERROR ("[pcl::NormalDistributionsTransform2D::computeTransformation] no overlap: try increasing the size or reducing the step of the grid\n");
      break;
    }

    transformPointCloud (output, intm_cloud, transformation);

    nr_iterations_++;
    
    if (update_visualizer_ != 0)
      update_visualizer_ (output, *indices_, *target_, *indices_);

    //std::cout << "eps=" << fabs ((transformation - previous_transformation_).sum ()) << std::endl;
    //std::cout << "curr eps_: " << (transformation - previous_transformation_).array ().abs ().sum () << std::endl;
    //std::cout << "nr_iterations_: " << nr_iterations_ << std::endl;

    if (nr_iterations_ > max_iterations_ ||
       (transformation - previous_transformation_).array ().abs ().sum () < transformation_epsilon_){
        converged_ = true;
        std::cout << " [NDT] ";
        std::cout << "iterations: " << nr_iterations_  << " final eps_: " << (transformation - previous_transformation_).array ().abs ().sum () << endl;// << " score.value: " << score.value;
        //cout << " xytheta: " << xytheta_transformation.transpose() << endl;
       }
  }

  final_transformation_ = transformation;
  output = intm_cloud;
}



template<typename PointSource, typename PointTarget> pcl::ndt2d::ValueAndDerivatives<3, double>
pcl::PWNormalDistributionsTransform2D<PointSource, PointTarget>::computeDerivatives(pcl::ndt2d::NDT2D<PointTarget> &target_ndt, PointCloudSource &intm_cloud, Eigen::Vector3d xytheta_transformation){
  pcl::ndt2d::ValueAndDerivatives<3, double> score = pcl::ndt2d::ValueAndDerivatives<3, double>::Zero ();
  for (size_t i = 0; i < intm_cloud.size (); i++){
    score += target_ndt.test (intm_cloud[i], std::cos (xytheta_transformation[2]), std::sin (xytheta_transformation[2]), bias_, gamma_);
  }
  return score;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointSource, typename PointTarget> bool
pcl::PWNormalDistributionsTransform2D<PointSource, PointTarget>::updateIntervalMT (double &a_l, double &f_l, double &g_l,
                                                                               double &a_u, double &f_u, double &g_u,
                                                                               double a_t, double f_t, double g_t)
{
  // Case U1 in Update Algorithm and Case a in Modified Update Algorithm [More, Thuente 1994]
  if (f_t > f_l)
  {
    a_u = a_t;
    f_u = f_t;
    g_u = g_t;
    return (false);
  }
  // Case U2 in Update Algorithm and Case b in Modified Update Algorithm [More, Thuente 1994]
  else
  if (g_t * (a_l - a_t) > 0)
  {
    a_l = a_t;
    f_l = f_t;
    g_l = g_t;
    return (false);
  }
  // Case U3 in Update Algorithm and Case c in Modified Update Algorithm [More, Thuente 1994]
  else
  if (g_t * (a_l - a_t) < 0)
  {
    a_u = a_l;
    f_u = f_l;
    g_u = g_l;

    a_l = a_t;
    f_l = f_t;
    g_l = g_t;
    return (false);
  }
  // Interval Converged
  else
    return (true);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointSource, typename PointTarget> double
pcl::PWNormalDistributionsTransform2D<PointSource, PointTarget>::trialValueSelectionMT (double a_l, double f_l, double g_l,
                                                                                    double a_u, double f_u, double g_u,
                                                                                    double a_t, double f_t, double g_t)
{
  // Case 1 in Trial Value Selection [More, Thuente 1994]
  if (f_t > f_l)
  {
    // Calculate the minimizer of the cubic that interpolates f_l, f_t, g_l and g_t
    // Equation 2.4.52 [Sun, Yuan 2006]
    double z = 3 * (f_t - f_l) / (a_t - a_l) - g_t - g_l;
    double w = std::sqrt (z * z - g_t * g_l);
    // Equation 2.4.56 [Sun, Yuan 2006]
    double a_c = a_l + (a_t - a_l) * (w - g_l - z) / (g_t - g_l + 2 * w);

    // Calculate the minimizer of the quadratic that interpolates f_l, f_t and g_l
    // Equation 2.4.2 [Sun, Yuan 2006]
    double a_q = a_l - 0.5 * (a_l - a_t) * g_l / (g_l - (f_l - f_t) / (a_l - a_t));

    if (std::fabs (a_c - a_l) < std::fabs (a_q - a_l))
      return (a_c);
    else
      return (0.5 * (a_q + a_c));
  }
  // Case 2 in Trial Value Selection [More, Thuente 1994]
  else
  if (g_t * g_l < 0)
  {
    // Calculate the minimizer of the cubic that interpolates f_l, f_t, g_l and g_t
    // Equation 2.4.52 [Sun, Yuan 2006]
    double z = 3 * (f_t - f_l) / (a_t - a_l) - g_t - g_l;
    double w = std::sqrt (z * z - g_t * g_l);
    // Equation 2.4.56 [Sun, Yuan 2006]
    double a_c = a_l + (a_t - a_l) * (w - g_l - z) / (g_t - g_l + 2 * w);

    // Calculate the minimizer of the quadratic that interpolates f_l, g_l and g_t
    // Equation 2.4.5 [Sun, Yuan 2006]
    double a_s = a_l - (a_l - a_t) / (g_l - g_t) * g_l;

    if (std::fabs (a_c - a_t) >= std::fabs (a_s - a_t))
      return (a_c);
    else
      return (a_s);
  }
  // Case 3 in Trial Value Selection [More, Thuente 1994]
  else
  if (std::fabs (g_t) <= std::fabs (g_l))
  {
    // Calculate the minimizer of the cubic that interpolates f_l, f_t, g_l and g_t
    // Equation 2.4.52 [Sun, Yuan 2006]
    double z = 3 * (f_t - f_l) / (a_t - a_l) - g_t - g_l;
    double w = std::sqrt (z * z - g_t * g_l);
    double a_c = a_l + (a_t - a_l) * (w - g_l - z) / (g_t - g_l + 2 * w);

    // Calculate the minimizer of the quadratic that interpolates g_l and g_t
    // Equation 2.4.5 [Sun, Yuan 2006]
    double a_s = a_l - (a_l - a_t) / (g_l - g_t) * g_l;

    double a_t_next;

    if (std::fabs (a_c - a_t) < std::fabs (a_s - a_t))
      a_t_next = a_c;
    else
      a_t_next = a_s;

    if (a_t > a_l)
      return (std::min (a_t + 0.66 * (a_u - a_t), a_t_next));
    else
      return (std::max (a_t + 0.66 * (a_u - a_t), a_t_next));
  }
  // Case 4 in Trial Value Selection [More, Thuente 1994]
  else
  {
    // Calculate the minimizer of the cubic that interpolates f_u, f_t, g_u and g_t
    // Equation 2.4.52 [Sun, Yuan 2006]
    double z = 3 * (f_t - f_u) / (a_t - a_u) - g_t - g_u;
    double w = std::sqrt (z * z - g_t * g_u);
    // Equation 2.4.56 [Sun, Yuan 2006]
    return (a_u + (a_t - a_u) * (w - g_u - z) / (g_t - g_u + 2 * w));
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointSource, typename PointTarget> double
pcl::PWNormalDistributionsTransform2D<PointSource, PointTarget>::computeStepLengthMT (const Eigen::Matrix<double, 3, 1> &x, Eigen::Matrix<double, 3, 1> &step_dir, double step_init, double step_max,
                                                                                  double step_min, ndt2d::ValueAndDerivatives<3, double> &score_all,
                                                                                  PointCloudSource &in_cloud, PointCloudSource &trans_cloud, pcl::ndt2d::NDT2D<PointTarget> &target_ndt)
{
  double score = score_all.value;
  Eigen::Matrix<double, 3, 1> score_gradient = score_all.grad;

  score = -score;
  score_gradient = -score_gradient;

  // Set the value of phi(0), Equation 1.3 [More, Thuente 1994]
  double phi_0 = -score;
  // Set the value of phi'(0), Equation 1.3 [More, Thuente 1994]
  double d_phi_0 = -(score_gradient.dot (step_dir));

  Eigen::Matrix<double, 3, 1>  x_t;

  if (d_phi_0 >= 0)
  {
    cout << "d_phi_0 >= 0 Reverse step direction " << endl;
    // Not a decent direction
    if (d_phi_0 == 0)
      return 0;
    else
    {
      // Reverse step direction and calculate optimal step.
      d_phi_0 *= -1;
      step_dir *= -1;

    }
  }

  // The Search Algorithm for T(mu) [More, Thuente 1994]

  int max_step_iterations = 10;
  int step_iterations = 0;

  // Sufficient decreace constant, Equation 1.1 [More, Thuete 1994]
  double mu = 1.e-4;
  // Curvature condition constant, Equation 1.2 [More, Thuete 1994]
  double nu = 0.9;

  // Initial endpoints of Interval I,
  double a_l = 0, a_u = 0;

  // Auxiliary function psi is used until I is determined ot be a closed interval, Equation 2.1 [More, Thuente 1994]
  double f_l = auxilaryFunction_PsiMT (a_l, phi_0, phi_0, d_phi_0, mu);
  double g_l = auxilaryFunction_dPsiMT (d_phi_0, d_phi_0, mu);

  double f_u = auxilaryFunction_PsiMT (a_u, phi_0, phi_0, d_phi_0, mu);
  double g_u = auxilaryFunction_dPsiMT (d_phi_0, d_phi_0, mu);

  // Check used to allow More-Thuente step length calculation to be skipped by making step_min == step_max
  bool interval_converged = (step_max - step_min) > 0, open_interval = true;

  double a_t = step_init;
  a_t = std::min (a_t, step_max);
  a_t = std::max (a_t, step_min);

  x_t = x + step_dir * a_t;

  final_transformation_ = (Eigen::Translation<float, 3>(static_cast<float> (x_t (0)), static_cast<float> (x_t (1)), static_cast<float> (0)) *
                           Eigen::AngleAxis<float> (static_cast<float> (x_t (2)), Eigen::Vector3f::UnitZ ())).matrix ();

  // New transformed point cloud
  //transformPointCloud (*input_, trans_cloud, final_transformation_);
  transformPointCloud (in_cloud, trans_cloud, final_transformation_);

  // Updates score, gradient and hessian.  Hessian calculation is unessisary but testing showed that most step calculations use the
  // initial step suggestion and recalculation the reusable portions of the hessian would intail more computation time.

  //score = computeDerivatives (score_gradient, hessian, trans_cloud, x_t, true); // [out]: score, score_gradient, hessian /////////////
  // Replace by my computeDerivatives
  score_all = computeDerivatives(target_ndt, trans_cloud, x_t);
  score = score_all.value;
  score_gradient = score_all.grad;

  score = -score;
  score_gradient = -score_gradient;

  // Calculate phi(alpha_t)
  double phi_t = -score;
  // Calculate phi'(alpha_t)
  double d_phi_t = -(score_gradient.dot (step_dir));

  // Calculate psi(alpha_t)
  double psi_t = auxilaryFunction_PsiMT (a_t, phi_t, phi_0, d_phi_0, mu);
  // Calculate psi'(alpha_t)
  double d_psi_t = auxilaryFunction_dPsiMT (d_phi_t, d_phi_0, mu);

  // Iterate until max number of iterations, interval convergance or a value satisfies the sufficient decrease, Equation 1.1, and curvature condition, Equation 1.2 [More, Thuente 1994]
  while (!interval_converged && step_iterations < max_step_iterations && !(psi_t <= 0 /*Sufficient Decrease*/ && d_phi_t <= -nu * d_phi_0 /*Curvature Condition*/))
  {
    cout << "computeStepLengthMT ! step_iterations: " << step_iterations << endl;
    // Use auxilary function if interval I is not closed
    if (open_interval)
    {
      a_t = trialValueSelectionMT (a_l, f_l, g_l,
                                   a_u, f_u, g_u,
                                   a_t, psi_t, d_psi_t);
    }
    else
    {
      a_t = trialValueSelectionMT (a_l, f_l, g_l,
                                   a_u, f_u, g_u,
                                   a_t, phi_t, d_phi_t);
    }

    cout << " current a_t: " << a_t << endl;

    a_t = std::min (a_t, step_max);
    a_t = std::max (a_t, step_min);

    x_t = x + step_dir * a_t;

    final_transformation_ = (Eigen::Translation<float, 3> (static_cast<float> (x_t (0)), static_cast<float> (x_t (1)), static_cast<float> (0)) *
                             Eigen::AngleAxis<float> (static_cast<float> (x_t (2)), Eigen::Vector3f::UnitZ ())).matrix ();

    // New transformed point cloud
    // Done on final cloud to prevent wasted computation
    //transformPointCloud (*input_, trans_cloud, final_transformation_);
    transformPointCloud (in_cloud, trans_cloud, final_transformation_);

    // Updates score, gradient. Values stored to prevent wasted computation.
    //score = computeDerivatives (score_gradient, hessian, trans_cloud, x_t, false); /////////////
    // Replace by my computeDerivatives
    score_all = computeDerivatives(target_ndt, trans_cloud, x_t);
    score = score_all.value;
    score_gradient = score_all.grad;

    score = -score;
    score_gradient = -score_gradient;

    // Calculate phi(alpha_t+)
    phi_t = -score;
    // Calculate phi'(alpha_t+)
    d_phi_t = -(score_gradient.dot (step_dir));

    // Calculate psi(alpha_t+)
    psi_t = auxilaryFunction_PsiMT (a_t, phi_t, phi_0, d_phi_0, mu);
    // Calculate psi'(alpha_t+)
    d_psi_t = auxilaryFunction_dPsiMT (d_phi_t, d_phi_0, mu);

    // Check if I is now a closed interval
    if (open_interval && (psi_t <= 0 && d_psi_t >= 0))
    {
      open_interval = false;

      // Converts f_l and g_l from psi to phi
      f_l = f_l + phi_0 - mu * d_phi_0 * a_l;
      g_l = g_l + mu * d_phi_0;

      // Converts f_u and g_u from psi to phi
      f_u = f_u + phi_0 - mu * d_phi_0 * a_u;
      g_u = g_u + mu * d_phi_0;
    }

    if (open_interval)
    {
      // Update interval end points using Updating Algorithm [More, Thuente 1994]
      interval_converged = updateIntervalMT (a_l, f_l, g_l,
                                             a_u, f_u, g_u,
                                             a_t, psi_t, d_psi_t);
    }
    else
    {
      // Update interval end points using Modified Updating Algorithm [More, Thuente 1994]
      interval_converged = updateIntervalMT (a_l, f_l, g_l,
                                             a_u, f_u, g_u,
                                             a_t, phi_t, d_phi_t);
    }

    step_iterations++;
  }

  // If inner loop was run then hessian needs to be calculated.
  // Hessian is unnessisary for step length determination but gradients are required
  // so derivative and transform data is stored for the next iteration.
//  if (step_iterations)
//     computeHessian (hessian, trans_cloud, x_t); ///////////// [TODO] In current version, we always compute hessian.

  //cout << "step_init: " << step_init << endl;
  //cout << "a_t: " << a_t << endl;

  return (a_t);
}


#endif    // PCL_NDT_2D_IMPL_H_
 
