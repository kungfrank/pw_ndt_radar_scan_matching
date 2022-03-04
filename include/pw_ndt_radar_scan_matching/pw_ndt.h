/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2011, Willow Garage, Inc.
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

#ifndef PCL_NDT_2D_H_
#define PCL_NDT_2D_H_

#include <pcl/registration/registration.h>

namespace pcl
{
  namespace ndt2d
  {
    /** \brief Class to store vector value and first and second derivatives
      * (grad vector and hessian matrix), so they can be returned easily from
      * functions
      */
    template<unsigned N=3, typename T=double>
    struct ValueAndDerivatives
    {
      ValueAndDerivatives () : hessian (), grad (), value () {}

      Eigen::Matrix<T, N, N> hessian;
      Eigen::Matrix<T, N, 1>    grad;
      T value;

      static ValueAndDerivatives<N,T>
      Zero ()
      {
        ValueAndDerivatives<N,T> r;
        r.hessian = Eigen::Matrix<T, N, N>::Zero ();
        r.grad    = Eigen::Matrix<T, N, 1>::Zero ();
        r.value   = 0;
        return r;
      }

      ValueAndDerivatives<N,T>&
      operator+= (ValueAndDerivatives<N,T> const& r)
      {
        hessian += r.hessian;
        grad    += r.grad;
        value   += r.value;
        return *this;
      }
    };

    /** \brief A normal distribution estimation class.
      *
      * First the indices of of the points from a point cloud that should be
      * modelled by the distribution are added with addIdx (...).
      *
      * Then estimateParams (...) uses the stored point indices to estimate the
      * parameters of a normal distribution, and discards the stored indices.
      *
      * Finally the distriubution, and its derivatives, may be evaluated at any
      * point using test (...).
      */
    template <typename PointT>
    class NormalDist
    {
      typedef pcl::PointCloud<PointT> PointCloud;

      public:
        NormalDist ()
          : min_n_ (4), n_ (0), pt_indices_ (), mean_ (), covar_inv_ ()
          // min_n_: 20 default, 80 for step 15 res .25, 4 for res .4, 190 for res .0432 ////////////////////////// TODO depend on grid size !!!!!!!!!!!
        {
        }

        /** \brief Store a point index to use later for estimating distribution parameters.
          * \param[in] i Point index to store
          */
        void
        addIdx (size_t i)
        {
          pt_indices_.push_back (i);
        }
        ///--- Visualization ---///
        void Vizsualization(bool viz,
                            const pcl::PointCloud<pcl::PointXYZI>::Ptr in_pc,
                            const pcl::PointCloud<pcl::PointXYZI>::Ptr output_cloud,
                            const pcl::PointCloud<pcl::PointXYZI>::Ptr old_in_pc){

            // Initializing point cloud visualizer
            pcl::visualization::PCLVisualizer::Ptr
            viewer_final (new pcl::visualization::PCLVisualizer ("3D Viewer"));
            viewer_final->setBackgroundColor (0.1, 0.1, 0.1);

            // Coloring and visualizing target cloud (red).
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> target_color (in_pc, 255, 0, 0);
            viewer_final->addPointCloud<pcl::PointXYZI> (in_pc, target_color, "target_cloud");

            // Coloring and visualizing transformed input cloud (green).
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> output_color (output_cloud, 0, 255, 0);
            viewer_final->addPointCloud<pcl::PointXYZI> (output_cloud, output_color, "output_cloud");

            // Coloring and visualizing transformed input cloud (blue).
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> input_color (old_in_pc, 0, 30, 255);
            viewer_final->addPointCloud<pcl::PointXYZI> (old_in_pc, input_color, "input_cloud");

            // Starting visualizer
            viewer_final->addCoordinateSystem (1.0, "global");
            viewer_final->initCameraParameters ();

            viewer_final->setCameraPosition (0,0,800,0,0,0,0,0,0);

            // Wait until visualizer window is closed.
            while (!viewer_final->wasStopped ())
            {
              viewer_final->spinOnce (100);
              std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        }
        /** \brief Estimate the normal distribution parameters given the point indices provided. Memory of point indices is cleared.
          * \param[in] cloud                    Point cloud corresponding to indices passed to addIdx.
          * \param[in] min_covar_eigvalue_mult  Set the smallest eigenvalue to this times the largest.
          */
        void
        estimateParams (const PointCloud& cloud, float& bias, float& gamma, const Eigen::Vector2f step, double min_covar_eigvalue_mult = 0.001)
        {
          Eigen::Vector2d sx  = Eigen::Vector2d::Zero ();
          double weight_sum = 0;
          Eigen::Matrix2d sxx = Eigen::Matrix2d::Zero ();

          std::vector<size_t>::const_iterator i;
          n_ = pt_indices_.size ();

          for (i = pt_indices_.begin (); i != pt_indices_.end (); i++)
          {
            Eigen::Vector2d p (cloud[*i]. x, cloud[*i]. y);

            double weight = cloud[*i].intensity - bias;
            sx += p * weight;
            //sx += p;
            weight_sum += weight;
            sxx += p * p.transpose () * weight;
            //sxx += p * p.transpose ();
          }

          if (n_ >= min_n_)
          {
            mean_ = sx / weight_sum;
            Eigen::Matrix2d covar = (sxx - 2 * (sx * mean_.transpose ())) / static_cast<double> (weight_sum) + mean_ * mean_.transpose ();
            //mean_ = sx / static_cast<double> (n_);
            //Eigen::Matrix2d covar = (sxx - 2 * (sx * mean_.transpose ())) / static_cast<double> (n_) + mean_ * mean_.transpose ();

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver (covar);
            if (solver.eigenvalues ()[0] < min_covar_eigvalue_mult * solver.eigenvalues ()[1])
            {
              PCL_DEBUG ("[pcl::NormalDist::estimateParams] NDT normal fit: adjusting eigenvalue %f\n", solver.eigenvalues ()[0]);
              Eigen::Matrix2d l = solver.eigenvalues ().asDiagonal ();
              Eigen::Matrix2d q = solver.eigenvectors ();
              // set minimum smallest eigenvalue:
              l (0,0) = l (1,1) * min_covar_eigvalue_mult;
              covar = q * l * q.transpose ();
            }
            covar_inv_ = covar.inverse ();
          }
          else{
            mean_ = Eigen::Vector2d::Zero ();
          }

          pt_indices_.clear ();
        }

        /** \brief Return the 'score' (denormalised likelihood) and derivatives of score of the point p given this distribution.
          * \param[in] transformed_pt   Location to evaluate at.
          * \param[in] cos_theta        sin(theta) of the current rotation angle of rigid transformation: to avoid repeated evaluation
          * \param[in] sin_theta        cos(theta) of the current rotation angle of rigid transformation: to avoid repeated evaluation
          * estimateParams must have been called after at least three points were provided, or this will return zero.
          *
          */
        ValueAndDerivatives<3,double>
        test (const PointT& transformed_pt, const double& cos_theta, const double& sin_theta, float& bias, float& gamma) const
        {
          if (n_ < min_n_)
            return ValueAndDerivatives<3,double>::Zero ();

          double n_weight = transformed_pt.intensity - bias;

          ValueAndDerivatives<3,double> r;
          const double x = transformed_pt.x;
          const double y = transformed_pt.y;
          const Eigen::Vector2d p_xy (transformed_pt.x, transformed_pt.y);
          const Eigen::Vector2d q = p_xy - mean_;
          const Eigen::RowVector2d qt_cvi (q.transpose () * covar_inv_);
          const double exp_qt_cvi_q = std::exp (-0.5 * double (qt_cvi * q));
          r.value = -exp_qt_cvi_q * n_weight; // ADD WEIGHT !!!!

          Eigen::Matrix<double, 2, 3> jacobian;
          jacobian <<
            1, 0, -(x * sin_theta + y*cos_theta),
            0, 1,   x * cos_theta - y*sin_theta;

          for (size_t i = 0; i < 3; i++)
            r.grad[i] = double (qt_cvi * jacobian.col (i)) * exp_qt_cvi_q * n_weight; // ADD WEIGHT !!!!

          // second derivative only for i == j == 2:
          const Eigen::Vector2d d2q_didj (
              y * sin_theta - x*cos_theta,
            -(x * sin_theta + y*cos_theta)
          );

          for (size_t i = 0; i < 3; i++)
            for (size_t j = 0; j < 3; j++)
              r.hessian (i,j) = -exp_qt_cvi_q * (
                double (-qt_cvi*jacobian.col (i)) * double (-qt_cvi*jacobian.col (j)) +
                (-qt_cvi * ((i==2 && j==2)? d2q_didj : Eigen::Vector2d::Zero ())) +
                (-jacobian.col (j).transpose () * covar_inv_ * jacobian.col (i))
              ) * n_weight; // ADD WEIGHT !!!!

          //cout << "r.value: " << r.value << "\n r.grad: \n" << r.grad << "\n r.hessian (i,j): \n" << r.hessian << endl;
          return r;
        }

        Eigen::Vector2d
        getMean(){
          return mean_;
        }

        Eigen::Matrix2d
        getCovar(){
          return covar_inv_.inverse();
        }

    protected:
        //const size_t min_n_;
        size_t min_n_;

        size_t n_;
        std::vector<size_t> pt_indices_;
        Eigen::Vector2d mean_;
        Eigen::Matrix2d covar_inv_;
    };

    /** \brief Build a set of normal distributions modelling a 2D point cloud,
      * and provide the value and derivatives of the model at any point via the
      * test (...) function.
      */
    template <typename PointT>
    class NDTSingleGrid: public boost::noncopyable
    {
      typedef typename pcl::PointCloud<PointT> PointCloud;
      typedef typename pcl::PointCloud<PointT>::ConstPtr PointCloudConstPtr;
      typedef typename pcl::ndt2d::NormalDist<PointT> NormalDist;

      public:
        NDTSingleGrid (PointCloudConstPtr cloud,
                       const Eigen::Vector2f& about,
                       const Eigen::Vector2f& extent,
                       const Eigen::Vector2f& step,
                       float& bias,
                       float& gamma)
            : min_ (about - extent), max_ (min_ + 2*extent), step_ (step),
              cells_ ((max_[0]-min_[0]) / step_[0],
                      (max_[1]-min_[1]) / step_[1]),
              normal_distributions_ (cells_[0], cells_[1])
        {
          // sort through all points, assigning them to distributions:
          NormalDist* n;
          size_t used_points = 0;
          for (size_t i = 0; i < cloud->size (); i++)
          if ((n = normalDistForPoint (cloud->at (i))))
          {
            n->addIdx (i);
            used_points++;
          }

          PCL_DEBUG ("[pcl::NDTSingleGrid] NDT single grid %dx%d using %d/%d points\n", cells_[0], cells_[1], used_points, cloud->size ());

          // then bake the distributions such that they approximate the
          // points (and throw away memory of the points)
          for (int x = 0; x < cells_[0]; x++)
            for (int y = 0; y < cells_[1]; y++)
              normal_distributions_.coeffRef (x,y).estimateParams (*cloud, bias, gamma, step);
        }

        Eigen::Matrix<NormalDist, Eigen::Dynamic, Eigen::Dynamic>
        get_normal_distributions (){
          return normal_distributions_;}

        /** \brief Return the 'score' (denormalised likelihood) and derivatives of score of the point p given this distribution.
          * \param[in] transformed_pt   Location to evaluate at.
          * \param[in] cos_theta        sin(theta) of the current rotation angle of rigid transformation: to avoid repeated evaluation
          * \param[in] sin_theta        cos(theta) of the current rotation angle of rigid transformation: to avoid repeated evaluation
          */
        ValueAndDerivatives<3,double>
        test (const PointT& transformed_pt, const double& cos_theta, const double& sin_theta, float& bias, float& gamma) const
        {
          const NormalDist* n = normalDistForPoint (transformed_pt);
          // index is in grid, return score from the normal distribution from
          // the correct part of the grid:
          if (n)
            return n->test (transformed_pt, cos_theta, sin_theta, bias, gamma);
          else
            return ValueAndDerivatives<3,double>::Zero ();
        }

      protected:
        /** \brief Return the normal distribution covering the location of point p
          * \param[in] p a point
          */
        NormalDist*
        normalDistForPoint (PointT const& p) const
        {
          // this would be neater in 3d...
          Eigen::Vector2f idxf;
          for (size_t i = 0; i < 2; i++)
            idxf[i] = (p.getVector3fMap ()[i] - min_[i]) / step_[i];
          Eigen::Vector2i idxi = idxf.cast<int> ();
          for (size_t i = 0; i < 2; i++)
            if (idxi[i] >= cells_[i] || idxi[i] < 0)
              return NULL;
          // const cast to avoid duplicating this function in const and
          // non-const variants...
          return const_cast<NormalDist*> (&normal_distributions_.coeffRef (idxi[0], idxi[1]));
        }

        Eigen::Vector2f min_;
        Eigen::Vector2f max_;
        Eigen::Vector2f step_;
        Eigen::Vector2i cells_;

        Eigen::Matrix<NormalDist, Eigen::Dynamic, Eigen::Dynamic> normal_distributions_;
    };

    /** \brief Build a Normal Distributions Transform of a 2D point cloud. This
      * consists of the sum of four overlapping models of the original points
      * with normal distributions.
      * The value and derivatives of the model at any point can be evaluated
      * with the test (...) function.
      */
    template <typename PointT>
    class NDT2D: public boost::noncopyable
    {
      typedef typename pcl::PointCloud<PointT> PointCloud;
      typedef typename pcl::PointCloud<PointT>::ConstPtr PointCloudConstPtr;
      typedef NDTSingleGrid<PointT> SingleGrid;

      typedef typename pcl::ndt2d::NormalDist<PointT> NormalDist;

      public:
        /** \brief
          * \param[in] cloud the input point cloud
          * \param[in] about Centre of the grid for normal distributions model
          * \param[in] extent Extent of grid for normal distributions model
          * \param[in] step Size of region that each normal distribution will model
          */
        NDT2D (PointCloudConstPtr cloud,
             const Eigen::Vector2f& about,
             const Eigen::Vector2f& extent,
             const Eigen::Vector2f& step,
             float& bias,
             float& gamma,
             Eigen::Matrix<NormalDist, Eigen::Dynamic, Eigen::Dynamic>& normal_distributions_matrix)
        {
          Eigen::Vector2f dx (step[0]/2, 0);
          Eigen::Vector2f dy (0, step[1]/2);
          single_grids_[0] = boost::make_shared<SingleGrid> (cloud, about,        extent, step, bias, gamma);
          single_grids_[1] = boost::make_shared<SingleGrid> (cloud, about +dx,    extent, step, bias, gamma);
          single_grids_[2] = boost::make_shared<SingleGrid> (cloud, about +dy,    extent, step, bias, gamma);
          single_grids_[3] = boost::make_shared<SingleGrid> (cloud, about +dx+dy, extent, step, bias, gamma);

          normal_distributions_matrix = single_grids_[0]->get_normal_distributions();
        }

        /** \brief Return the 'score' (denormalised likelihood) and derivatives of score of the point p given this distribution.
          * \param[in] transformed_pt   Location to evaluate at.
          * \param[in] cos_theta        sin(theta) of the current rotation angle of rigid transformation: to avoid repeated evaluation
          * \param[in] sin_theta        cos(theta) of the current rotation angle of rigid transformation: to avoid repeated evaluation
          */
        ValueAndDerivatives<3,double>
        test (const PointT& transformed_pt, const double& cos_theta, const double& sin_theta, float& bias, float& gamma) const
        {
          ValueAndDerivatives<3,double> r = ValueAndDerivatives<3,double>::Zero ();
          for (size_t i = 0; i < 4; i++)
              r += single_grids_[i]->test (transformed_pt, cos_theta, sin_theta, bias, gamma);
          return r;
        }

      protected:
        boost::shared_ptr<SingleGrid> single_grids_[4];
    };

  } // namespace ndt2d


  /** \brief @b NormalDistributionsTransform2D provides an implementation of the
    * Normal Distributions Transform algorithm for scan matching.
    *
    * This implementation is intended to match the definition:
    * Peter Biber and Wolfgang Straßer. The normal distributions transform: A
    * new approach to laser scan matching. In Proceedings of the IEEE In-
    * ternational Conference on Intelligent Robots and Systems (IROS), pages
    * 2743–2748, Las Vegas, USA, October 2003.
    *
    * \author James Crosby
    */
  template <typename PointSource, typename PointTarget>
  class PWNormalDistributionsTransform2D : public Registration<PointSource, PointTarget>
  {
    typedef typename Registration<PointSource, PointTarget>::PointCloudSource PointCloudSource;
    typedef typename PointCloudSource::Ptr PointCloudSourcePtr;
    typedef typename PointCloudSource::ConstPtr PointCloudSourceConstPtr;

    typedef typename Registration<PointSource, PointTarget>::PointCloudTarget PointCloudTarget;

    typedef PointIndices::Ptr PointIndicesPtr;
    typedef PointIndices::ConstPtr PointIndicesConstPtr;

    public:

        typedef boost::shared_ptr< PWNormalDistributionsTransform2D<PointSource, PointTarget> > Ptr;
        typedef boost::shared_ptr< const PWNormalDistributionsTransform2D<PointSource, PointTarget> > ConstPtr;

      /** \brief Empty constructor. */
      PWNormalDistributionsTransform2D ()
        : Registration<PointSource,PointTarget> (),
          grid_centre_ (0,0), grid_step_ (1,1), grid_extent_ (20,20), newton_lambda_ (1,1,1), max_step_size_ (0.01), bias_(70), gamma_(1)
      {
        reg_name_ = "PWNormalDistributionsTransform2D";
      }

      /** \brief Empty destructor */
      virtual ~PWNormalDistributionsTransform2D () {}

      /** \brief centre of the ndt grid (target coordinate system)
        * \param centre value to set
        */
      virtual void
      setGridCentre (const Eigen::Vector2f& centre) { grid_centre_ = centre; }

      /** \brief Grid spacing (step) of the NDT grid
        * \param[in] step value to set
        */
      virtual void
      setGridStep (const Eigen::Vector2f& step) { grid_step_ = step; }

      /** \brief NDT Grid extent (in either direction from the grid centre)
        * \param[in] extent value to set
        */
      virtual void
      setGridExtent (const Eigen::Vector2f& extent) { grid_extent_ = extent; }

      /** \brief NDT Newton optimisation step size parameter
        * \param[in] lambda step size: 1 is simple newton optimisation, smaller values may improve convergence
        */
      virtual void
      setOptimizationStepSize (const double& lambda) { newton_lambda_ = Eigen::Vector3d (lambda, lambda, lambda); }

      /** \brief NDT Newton optimisation step size parameter
        * \param[in] lambda step size: (1,1,1) is simple newton optimisation,
        * smaller values may improve convergence, or elements may be set to
        * zero to prevent optimisation over some parameters
        *
        * This overload allows control of updates to the individual (x, y,
        * theta) free parameters in the optimisation. If, for example, theta is
        * believed to be close to the correct value a small value of lambda[2]
        * should be used.
        */
      virtual void
      setOptimizationStepSize (const Eigen::Vector3d& lambda) { newton_lambda_ = lambda; }

      virtual void
      setMaxStepSize (const double& max_step_size) { max_step_size_ = max_step_size; }

      virtual void
      setIntensityBias (const float& bias) { bias_ = bias; }

      Eigen::Matrix< pcl::ndt2d::NormalDist<PointTarget> , Eigen::Dynamic, Eigen::Dynamic>
      getNormalDistributionMap () {
        return normal_distributions_map_;
      }

    protected:
      /** \brief Rigid transformation computation method with initial guess.
        * \param[out] output the transformed input point cloud dataset using the rigid transformation found
        * \param[in] guess the initial guess of the transformation to compute
        */
      virtual void
      computeTransformation (PointCloudSource &output, const Eigen::Matrix4f &guess);

      using Registration<PointSource, PointTarget>::reg_name_;
      using Registration<PointSource, PointTarget>::target_;
      using Registration<PointSource, PointTarget>::converged_;
      using Registration<PointSource, PointTarget>::nr_iterations_;
      using Registration<PointSource, PointTarget>::max_iterations_;
      using Registration<PointSource, PointTarget>::transformation_epsilon_;
      using Registration<PointSource, PointTarget>::transformation_;
      using Registration<PointSource, PointTarget>::previous_transformation_;
      using Registration<PointSource, PointTarget>::final_transformation_;
      using Registration<PointSource, PointTarget>::update_visualizer_;
      using Registration<PointSource, PointTarget>::indices_;


      pcl::ndt2d::ValueAndDerivatives<3, double>
      computeDerivatives(pcl::ndt2d::NDT2D<PointTarget> &target_ndt, PointCloudSource &intm_cloud, Eigen::Vector3d xytheta_transformation);

      /** \brief Compute line search step length and update transform and probability derivatives using More-Thuente method.
        * \note Search Algorithm [More, Thuente 1994]
        * \param[in] x initial transformation vector, \f$ x \f$ in Equation 1.3 (Moore, Thuente 1994) and \f$ \vec{p} \f$ in Algorithm 2 [Magnusson 2009]
        * \param[in] step_dir descent direction, \f$ p \f$ in Equation 1.3 (Moore, Thuente 1994) and \f$ \delta \vec{p} \f$ normalized in Algorithm 2 [Magnusson 2009]
        * \param[in] step_init initial step length estimate, \f$ \alpha_0 \f$ in Moore-Thuente (1994) and the noramal of \f$ \delta \vec{p} \f$ in Algorithm 2 [Magnusson 2009]
        * \param[in] step_max maximum step length, \f$ \alpha_max \f$ in Moore-Thuente (1994)
        * \param[in] step_min minimum step length, \f$ \alpha_min \f$ in Moore-Thuente (1994)
        * \param[out] score final score function value, \f$ f(x + \alpha p) \f$ in Equation 1.3 (Moore, Thuente 1994) and \f$ score \f$ in Algorithm 2 [Magnusson 2009]
        * \param[in,out] score_gradient gradient of score function w.r.t. transformation vector, \f$ f'(x + \alpha p) \f$ in Moore-Thuente (1994) and \f$ \vec{g} \f$ in Algorithm 2 [Magnusson 2009]
        * \param[out] hessian hessian of score function w.r.t. transformation vector, \f$ f''(x + \alpha p) \f$ in Moore-Thuente (1994) and \f$ H \f$ in Algorithm 2 [Magnusson 2009]
        * \param[in,out] trans_cloud transformed point cloud, \f$ X \f$ transformed by \f$ T(\vec{p},\vec{x}) \f$ in Algorithm 2 [Magnusson 2009]
        * \return final step length
        */
      double
      computeStepLengthMT (const Eigen::Matrix<double, 3, 1> &x,
                           Eigen::Matrix<double, 3, 1> &step_dir,
                           double step_init,
                           double step_max, double step_min,
                           pcl::ndt2d::ValueAndDerivatives<3, double> &score,
                           PointCloudSource &in_cloud,
                           PointCloudSource &trans_cloud,
                           pcl::ndt2d::NDT2D<PointTarget> &target_ndt);

      /** \brief Update interval of possible step lengths for More-Thuente method, \f$ I \f$ in More-Thuente (1994)
        * \note Updating Algorithm until some value satifies \f$ \psi(\alpha_k) \leq 0 \f$ and \f$ \phi'(\alpha_k) \geq 0 \f$
        * and Modified Updating Algorithm from then on [More, Thuente 1994].
        * \param[in,out] a_l first endpoint of interval \f$ I \f$, \f$ \alpha_l \f$ in Moore-Thuente (1994)
        * \param[in,out] f_l value at first endpoint, \f$ f_l \f$ in Moore-Thuente (1994), \f$ \psi(\alpha_l) \f$ for Update Algorithm and \f$ \phi(\alpha_l) \f$ for Modified Update Algorithm
        * \param[in,out] g_l derivative at first endpoint, \f$ g_l \f$ in Moore-Thuente (1994), \f$ \psi'(\alpha_l) \f$ for Update Algorithm and \f$ \phi'(\alpha_l) \f$ for Modified Update Algorithm
        * \param[in,out] a_u second endpoint of interval \f$ I \f$, \f$ \alpha_u \f$ in Moore-Thuente (1994)
        * \param[in,out] f_u value at second endpoint, \f$ f_u \f$ in Moore-Thuente (1994), \f$ \psi(\alpha_u) \f$ for Update Algorithm and \f$ \phi(\alpha_u) \f$ for Modified Update Algorithm
        * \param[in,out] g_u derivative at second endpoint, \f$ g_u \f$ in Moore-Thuente (1994), \f$ \psi'(\alpha_u) \f$ for Update Algorithm and \f$ \phi'(\alpha_u) \f$ for Modified Update Algorithm
        * \param[in] a_t trial value, \f$ \alpha_t \f$ in Moore-Thuente (1994)
        * \param[in] f_t value at trial value, \f$ f_t \f$ in Moore-Thuente (1994), \f$ \psi(\alpha_t) \f$ for Update Algorithm and \f$ \phi(\alpha_t) \f$ for Modified Update Algorithm
        * \param[in] g_t derivative at trial value, \f$ g_t \f$ in Moore-Thuente (1994), \f$ \psi'(\alpha_t) \f$ for Update Algorithm and \f$ \phi'(\alpha_t) \f$ for Modified Update Algorithm
        * \return if interval converges
        */
      bool
      updateIntervalMT (double &a_l, double &f_l, double &g_l,
                        double &a_u, double &f_u, double &g_u,
                        double a_t, double f_t, double g_t);

      /** \brief Select new trial value for More-Thuente method.
        * \note Trial Value Selection [More, Thuente 1994], \f$ \psi(\alpha_k) \f$ is used for \f$ f_k \f$ and \f$ g_k \f$
        * until some value satifies the test \f$ \psi(\alpha_k) \leq 0 \f$ and \f$ \phi'(\alpha_k) \geq 0 \f$
        * then \f$ \phi(\alpha_k) \f$ is used from then on.
        * \note Interpolation Minimizer equations from Optimization Theory and Methods: Nonlinear Programming By Wenyu Sun, Ya-xiang Yuan (89-100).
        * \param[in] a_l first endpoint of interval \f$ I \f$, \f$ \alpha_l \f$ in Moore-Thuente (1994)
        * \param[in] f_l value at first endpoint, \f$ f_l \f$ in Moore-Thuente (1994)
        * \param[in] g_l derivative at first endpoint, \f$ g_l \f$ in Moore-Thuente (1994)
        * \param[in] a_u second endpoint of interval \f$ I \f$, \f$ \alpha_u \f$ in Moore-Thuente (1994)
        * \param[in] f_u value at second endpoint, \f$ f_u \f$ in Moore-Thuente (1994)
        * \param[in] g_u derivative at second endpoint, \f$ g_u \f$ in Moore-Thuente (1994)
        * \param[in] a_t previous trial value, \f$ \alpha_t \f$ in Moore-Thuente (1994)
        * \param[in] f_t value at previous trial value, \f$ f_t \f$ in Moore-Thuente (1994)
        * \param[in] g_t derivative at previous trial value, \f$ g_t \f$ in Moore-Thuente (1994)
        * \return new trial value
        */
      double
      trialValueSelectionMT (double a_l, double f_l, double g_l,
                             double a_u, double f_u, double g_u,
                             double a_t, double f_t, double g_t);

      /** \brief Auxilary function used to determin endpoints of More-Thuente interval.
        * \note \f$ \psi(\alpha) \f$ in Equation 1.6 (Moore, Thuente 1994)
        * \param[in] a the step length, \f$ \alpha \f$ in More-Thuente (1994)
        * \param[in] f_a function value at step length a, \f$ \phi(\alpha) \f$ in More-Thuente (1994)
        * \param[in] f_0 initial function value, \f$ \phi(0) \f$ in Moore-Thuente (1994)
        * \param[in] g_0 initial function gradiant, \f$ \phi'(0) \f$ in More-Thuente (1994)
        * \param[in] mu the step length, constant \f$ \mu \f$ in Equation 1.1 [More, Thuente 1994]
        * \return sufficent decrease value
        */
      inline double
      auxilaryFunction_PsiMT (double a, double f_a, double f_0, double g_0, double mu = 1.e-4)
      {
        return (f_a - f_0 - mu * g_0 * a);
      }

      /** \brief Auxilary function derivative used to determin endpoints of More-Thuente interval.
        * \note \f$ \psi'(\alpha) \f$, derivative of Equation 1.6 (Moore, Thuente 1994)
        * \param[in] g_a function gradient at step length a, \f$ \phi'(\alpha) \f$ in More-Thuente (1994)
        * \param[in] g_0 initial function gradiant, \f$ \phi'(0) \f$ in More-Thuente (1994)
        * \param[in] mu the step length, constant \f$ \mu \f$ in Equation 1.1 [More, Thuente 1994]
        * \return sufficent decrease derivative
        */
      inline double
      auxilaryFunction_dPsiMT (double g_a, double g_0, double mu = 1.e-4)
      {
        return (g_a - mu * g_0);
      }

      Eigen::Vector2f grid_centre_;
      Eigen::Vector2f grid_step_;
      Eigen::Vector2f grid_extent_;
      Eigen::Vector3d newton_lambda_;
      double max_step_size_;

      float bias_;
      float gamma_;

      Eigen::Matrix< pcl::ndt2d::NormalDist<PointTarget>, Eigen::Dynamic, Eigen::Dynamic> normal_distributions_map_;

    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
}

#include "pw_ndt_search_step.hpp"

#endif // ndef PCL_NDT_2D_H_

