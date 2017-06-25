#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

// Unscented Kalman Filter Project
// Creig Cavanaugh
// June 2017
// Rev 1: Updated per 6/17/17 Code review
// Rev 2: Updated per second 6/17/17 code review

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2.2;  

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.3;  

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  //set state dimension
  n_x_ = 5;

  //set augmented dimension
  n_aug_ = 7;

  //define spreading parameter
  lambda_ = 3 - n_aug_;

  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  //set measurement dimension, radar can measure r, phi, and r_dot
  n_z_radar_ = 3;

  //set measurement dimension, lidar can measure x, y
  n_z_laser_ = 2;

  //add measurement noise covariance matrix
  R_radar_ = MatrixXd(n_z_radar_,n_z_radar_);
  R_laser_ = MatrixXd(n_z_laser_,n_z_laser_);

  is_initialized_ = false;

  // NIS for radar
  NIS_radar_ = 0;

  // NIS for laser
  NIS_laser_ = 0;

  use_laser_ = true;

  use_radar_ = true;

  H_ = MatrixXd(2, 5);
  Ht_ = MatrixXd(5, 2);

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

  /************************************************
  *                 Initialization                *
  *************************************************/


  if (!is_initialized_) {

   previous_t_ = meas_package.timestamp_;

   //Updated per second 6/17/17 code review
   P_ <<    1, 0,  0,  0,  0,
   0,  1, 0,  0,  0,
   0,  0,  1, 0,  0,
   0,  0,  0,  1, 0,
   0,  0,  0,  0, 1;

   Xsig_pred_.fill(0.0);

   H_ <<     1, 0, 0, 0, 0,
   0, 1, 0, 0, 0;

   Ht_ = H_.transpose();


  //Measurement noise covariance
   R_laser_ <<   std_laspx_*std_laspx_, 0,
   0, std_laspy_*std_laspy_;

   R_radar_ <<   std_radr_*std_radr_, 0, 0,
   0, std_radphi_*std_radphi_, 0,
   0, 0,std_radrd_*std_radrd_;

   if ((meas_package.sensor_type_ == MeasurementPackage::RADAR) && (use_radar_)) {
    /**
    Convert radar from polar to cartesian coordinates and initialize state.
    ro,theta, ro_dot
    */

    //set the state with the initial location and zero velocity
    float rho = meas_package.raw_measurements_[0];
    float phi = meas_package.raw_measurements_[1];
    float rho_dot = meas_package.raw_measurements_[2];

    float x = rho * cos(phi);
    float y = rho * sin(phi);

    //Updated per 6/17/17 code review
    if ( fabs(x) < 0.001 && fabs(y) < 0.001 ) {
      x = 0.001;    
      y = 0.001;    
    }

    //Initialize state
    x_ << x, y, rho_dot, 0, 0;

  }
  else if ((meas_package.sensor_type_ == MeasurementPackage::LASER) && (use_laser_)) {
    /**
    Initialize state.
    px, py
    */
    float x = meas_package.raw_measurements_[0];
    float y = meas_package.raw_measurements_[1];

    //Updated per 6/17/17 code review
    if ( fabs(x) < 0.001 && fabs(y) < 0.001 ) {
      x = 0.001;    
      y = 0.001;    
    }

    //Initialize state
    x_ << x, y, 0, 0, 0;

  }
  else {
    return;
  }

  // done initializing, no need to predict or update
  is_initialized_ = true;

  return;
}

  /************************************************
  *                Control Structure              *
  *************************************************/

  /**********************
   *  Prediction
   **********************/
  //compute the time elapsed between the current and previous measurements
  float delta_t = (meas_package.timestamp_ - previous_t_) / 1000000.0; //dt - expressed in seconds
  previous_t_ = meas_package.timestamp_;

  Prediction(delta_t);


/*****************************************************************************
 *  Update
 ****************************************************************************/

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
  // Radar updates
    if (use_radar_){
      UpdateRadar(meas_package);
    }

  } else {
  // Laser updates
    if (use_laser_) {
      UpdateLidar(meas_package);
    }
  }

// print the output
  cout << "x_ = " << x_ << endl;
  cout << "P_ = " << P_ << endl;
  cout << "Radar NIS = " << NIS_radar_ << endl;
  //cout << "Laser NIS = " << NIS_laser_ << endl;  
}



/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {

  /**************************************************************
  *                 Create Augmented Sigma Points               *
  ***************************************************************/

  //Lesson 7, section 18: Augmentation Assignment 2

  //create augmented mean vector
  VectorXd x_aug = VectorXd(7);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(7, 7);

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  Xsig_aug.fill(0);

  //create augmented mean state
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  //create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_*std_a_;
  P_aug(6,6) = std_yawdd_*std_yawdd_;

  //create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  //create augmented sigma points
  Xsig_aug.col(0)  = x_aug;
  for (int i = 0; i< n_aug_; i++)
  {
    Xsig_aug.col(i+1)       = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
  }

  /**************************************************************
  *                     Predict Sigma Points                    *
  ***************************************************************/
  //Lesson 7, section 21: Sigma Point Prediction Assignment 2

  Xsig_pred_.fill(0);

  //predict sigma points
  for (int i = 0; i< 2*n_aug_+1; i++)
  {
    //extract values for better readability
    const double p_x = Xsig_aug(0,i);
    const double p_y = Xsig_aug(1,i);
    const double v = Xsig_aug(2,i);
    const double yaw = Xsig_aug(3,i);
    const double yawd = Xsig_aug(4,i);
    const double nu_a = Xsig_aug(5,i);
    const double nu_yawdd = Xsig_aug(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
      px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
      py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    }
    else {
      px_p = p_x + v*delta_t*cos(yaw);
      py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    //write predicted sigma point into right column

    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;

  }


  /**************************************************************
  *                     Predict mean and covariance             *
  ***************************************************************/
  //Lesson 7, section 24: Predicted Mean and Covariance Assignment 2

  weights_ = VectorXd(2*n_aug_+1);

 // set weights_
  double weight_0 = lambda_/(lambda_+n_aug_);
  weights_(0) = weight_0;
  for (int i=1; i<2*n_aug_+1; i++) {  //2n+1 weights
    double weight = 0.5/(n_aug_+lambda_);
    weights_(i) = weight;
  }

  //predicted state mean
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
    x_ = x_+ weights_(i) * Xsig_pred_.col(i);
  }

  //predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    //angle normalization
    // Updated per second 6/17/17 code review
    NormalizeAngle(x_diff(3));

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
  }

}






/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  //Reverted back to simplified kalman filter based on second 6/17/17 code review feedback

  VectorXd z = meas_package.raw_measurements_;
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  MatrixXd PHt = P_ * Ht_;
  MatrixXd S = H_ * PHt + R_laser_;
  MatrixXd Si = S.inverse();
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;

}


/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */

  //Lesson 7, section 27: Predict Radar Measurement Assignment 2

  VectorXd z = meas_package.raw_measurements_;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z_radar_, 2 * n_aug_ + 1);

  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    // extract values for better readibility
    double p_x = Xsig_pred_(0,i);

    double p_y = Xsig_pred_(1,i);

    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    //Updated per 6/17/17 code review
    //Check for divide by zero
    if ( fabs(p_x) < 0.001 && fabs(p_y) < 0.001  ){
      p_x = 0.001;
      p_y = 0.001;
    }

    // measurement model
    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
    Zsig(1,i) = atan2(p_y,p_x);                                 //phi
    Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot

  }

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z_radar_);
  z_pred.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; i++) {
    z_pred += weights_(i) * Zsig.col(i);
  }

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z_radar_,n_z_radar_);
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    //angle normalization
    // Updated per second 6/17/17 code review
    NormalizeAngle(z_diff(1));

    S += weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  S += R_radar_;

  /***********************************************
  *                     Update Radar             *
  ************************************************/

  //Lesson 7, section 30: UKF Update Assignment 2
  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z_radar_);

  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    //angle normalization
    // Updated per second 6/17/17 code review
    NormalizeAngle(z_diff(1));


    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    //angle normalization
    // Updated per second 6/17/17 code review
    NormalizeAngle(x_diff(3));

    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = z - z_pred;

  //angle normalization
  // Updated per second 6/17/17 code review
  NormalizeAngle(z_diff(1));

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();

  //NIS
  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;

}

//Normalize Angle
// Function added per second 6/17/17 code review
void UKF::NormalizeAngle(double& phi)
{
  phi = atan2(sin(phi), cos(phi));
}