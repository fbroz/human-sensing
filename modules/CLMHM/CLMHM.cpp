/**
 * This code is modified for the iCub humanoid robot through YARP connection.
 * The program reads the image from one camera (robot's eye) and tries to detect a face and exports the result on a new view.
 * It also sends the estimated pose of the head to
 * 
 * 
 * 
 * output:
 *  1- Sends the 3D pose of the Gaze to the port /clmgaze/3dpose/out
 * this can be used to connect to the iKinGazeControl  module
 *
 * 2- Sends a bottle including the corners of a bounding box and the center of two eyes
 * this can be used as an input for the face recognition module (for Vadim)
 * 
 * 
 * by: Reza Ahmadzadeh (reza.ahmadzadeh@iit.it)
 **/
// reference for YARP
#include <yarp/sig/Image.h>
#include <yarp/os/RFModule.h>
#include <yarp/os/Module.h>
#include <yarp/os/Network.h>
#include <yarp/sig/Vector.h>
#include <yarp/os/Bottle.h>
#include <yarp/dev/all.h>
#include <yarp/dev/Drivers.h>
#include <yarp/dev/ControlBoardInterfaces.h>
#include <yarp/dev/GazeControl.h>
#include <yarp/dev/PolyDriver.h>
#include <yarp/math/Math.h>
#include <yarp/os/Time.h>
#include <yarp/os/RateThread.h>
// opencv
/**
   #include <cv.h>
   #include <highgui.h>
**/
#include <opencv2/videoio/videoio.hpp>  // Video write
#include <opencv2/videoio/videoio_c.h>  // Video write
// c++
/**
   #include <iostream>
   #include <algorithm>
   #include <stdio.h>
   #include <stdlib.h>
   #include <cstdio>
**/
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <random>
#include <time.h>
// CLM library
/**
   #include <CLM.h>
   #include <CLMTracker.h>
   #include <CLMParameters.h>
   #include <CLM_utils.h>
**/
#include <CLM_core.h>
// others
#include <gsl/gsl_math.h>
#include <deque>
YARP_DECLARE_PLUGINS(icubmod)

/**
   #define CTRL_THREAD_PER 0.02 // [s]
   #define PRINT_STATUS_PER 1.0 // [s]
   #define STORE_POI_PER 3.0 // [s]
   #define SWITCH_STATE_PER 10.0 // [s]
   #define STILL_STATE_TIME 5.0 // [s]
   #define STATE_TRACK 0
   #define STATE_RECALL 1
   #define STATE_WAIT 2
   #define STATE_STILL 3
**/

#define CYCLING_TIME 10.0 // the time elapsed between each gaze switching in not real seconds

// --------------------------------------------------------------
// ------------------
// ----- Macros -----
// ------------------
// --------------------------------------------------------------
#define INFO_STREAM( stream )			\
  std::cout << stream << std::endl

#define WARN_STREAM( stream )				\
  std::cout << "Warning: " << stream << std::endl

#define ERROR_STREAM( stream )			\
  std::cout << "Error: " << stream << std::endl

  static void printErrorAndAbort( const std::string & error )
{
  std::cout << error << std::endl;
  abort();
}

//returns difference in seconds
double diffclock(clock_t clock1,clock_t clock2)
{
  double diffticks=clock1-clock2;
  double diffs=diffticks/CLOCKS_PER_SEC;
  return diffs;
}


const std::string currentDateTime() {
  time_t now = time(0);
  struct tm tstruct;
  char buf[80];
  tstruct = *localtime(&now);
  strftime(buf,sizeof(buf),"%Y-%m-%d.%H.%M.%S", &tstruct);
  return buf;
}


#define FATAL_STREAM( stream )					\
  printErrorAndAbort( std::string( "Fatal error: " ) + stream )






// --------------------------------------------------------------
// ----------------------
// ----- Namespaces -----
// ----------------------
// --------------------------------------------------------------
using namespace std;
using namespace cv;
using namespace yarp::os;
using namespace yarp::sig;
using namespace yarp::dev;
using namespace yarp::math;

// --------------------------------------------------------------
// -----------------
// ----- Class -----
// -----------------
// --------------------------------------------------------------
class MyModule:public RFModule
{

private:

  //Mat cvImage;
  //BufferedPort<Bottle> inPort, outPort;
	
  BufferedPort<ImageOf<PixelRgb> > imageIn;  // make a port for reading images
  BufferedPort<ImageOf<PixelRgb> > imageOut; // make a port for passing the result to
  BufferedPort<Bottle> targetPort; // for Vadim

  BufferedPort<Bottle> posePort; // for Ali
	
  IplImage* cvImage;
  IplImage* display;

  Mat_<float> depth_image;
  Mat_<uchar> grayscale_image;
  Mat captured_image;
  Mat_<short> depth_image_16_bit;
  Mat_<double> shape_3D;

		
  vector<string> arguments;					// The arguments from the input 
  vector<string> files, depth_directories, pose_output_files, tracked_videos_output, landmark_output_files, landmark_3D_output_files;

  bool use_camera_plane_pose, done, cx_undefined, use_depth, write_settings, ok2, gazeControl;
  bool runGaze; // set this value to true to run gaze control on the icub
  int device, f_n, frame_count;
  int state, startup_context_id, jnt;
  int thickness;
  int gazeChangeFlag, oldGazeChangeFlag;
  int gazeMode;
  
  float fx,fy,cx,cy; 							// parameters of the camera

  string current_file;

  double fps;
  double visualisation_boundary;
  double detection_certainty;

  int64 t1,t0;								// time parameters for calculating and storing the framerate of the camera

  CLMTracker::CLMParameters* clm_parameters;
  CLMTracker::CLM* clm_model;

  std::ofstream facestate_output_file;
  std::ofstream pose_output_file;
  std::ofstream landmarks_output_file;	
  std::ofstream landmarks_3D_output_file;
	
  PolyDriver clientGazeCtrl;
  IGazeControl *igaze;
  IPositionControl *pos;
  IEncoders *encs;
  IVelocityControl *vel;

  Vector tmp, position, command, encoders, velocity, acceleration;
  Vector fp, x;
  Vec6d pose_estimate_CLM, pose_estimate_to_draw;

  // parameters for transfromation of pose w.r.t eye to pose w.r.t root
  Vector pose_act, ori_act;       // actual pose and actual orientation of the left eye of icub
  Vector pose_clm, pose_robot;    // estimated pose by clm, caculated pose w.r.t the root of the robot
  Vector pose_clm_left_corner, pose_clm_right_corner; // for storing the data for the corners of the eyes
  Vector pose_left_eye, pose_right_eye, pose_mouth; //for storing facial feature positions
  Matrix H;                       // transformation matrx

  //bool modelControl;         //whether to use Markov chain to pick gaze target
  //std::ifstream model_control_file; //the file that contains the control matrix

  //std::default_random_engine generator;
  //std::normal_distribution<double> distribution;
  //std::random_device rd;
  //std::mt19937 generatorUniform;
  //std::uniform_int_distribution<> disUniform;

  clock_t beginTime, endTime;
  double timeToKeepAGaze;
  bool oneIter;


  enum GazeMode {fixed, random, model};
  const int NUM_GAZE_STATES = 7;
  //state order must be same as the order used for the discrete distributions
  enum GazeState {none, mouth, lefteye, righteye, left, right, up, down};

public:

  MyModule(){} 							// constructor
  ~MyModule(){}							// deconstructor


  double getPeriod()						// the period of the loop.
  {
    return 0.0; 						// 0 is equal to the real-time
  }


  // --------------------------------------------------------------
  bool configure(yarp::os::ResourceFinder &rf)
  {


    ConstString gmode = rf.find("mode").asString();
    if(gmode == "model") {
      cout << "Using model-based gaze" << endl;
      gazeMode = model;
    } else if(gmode == "random") {
      cout << "Using random gaze" << endl;
	gazeMode = random;
    } else if(gmode == "fixed") {
	cout << "Using fixed gaze" << endl;
	gazeMode = fixed;
    } else {
      cout << "No gaze mode specified, using fixed gaze (model, random, fixed) possible" << endl;
      gazeMode = fixed;
    }
    
    // --- open the ports ---
    ok2 = imageIn.open("/clmgaze/image/in");
    ok2 = ok2 && imageOut.open("/clmgaze/image/out");
    if (!ok2)
      {
	fprintf(stderr, "Error. Failed to open image ports. \n");
	return false;
      }	

    ok2 = targetPort.open("/clmgaze/cornerPose/out");
    if (!ok2)
      {
	fprintf(stderr,"Error. failed to open a port for the pose. \n");
	return false;
      }

    ok2 = posePort.open("/clmgaze/centerPose/out");
    if (!ok2)
      {
	fprintf(stderr,"Error. failed to open a port for the 3D pose. \n");
	return false;
      }
    // ---

    arguments.push_back("-f");							// provide two default arguments in case we want to use no real camera
    arguments.push_back("../../videos/default.wmv"); 	// the video file 
    device = 0;   										// camera


    // --- camera parameters: HW icub, Black - Blue, Blue 640x480, others
		fx = 234.519; //232.921; //225.904; //443; //211; //225.904; //409.9; //225.904; //500; (default)
		fy = 234.158;//232.43; //227.041; //444; //211; //227.041;//409.023; //227.041; //500; (default)
		cx = 161.548;//162.91; //157.875; //344; //161; //157.858; //337.575; //157.858; //0; (default)
		cy = 153.602;//125.98; //113.51; //207; //128; //113.51; //250.798; //113.51; //0; (default)

    clm_parameters = new CLMTracker::CLMParameters(arguments);

    // checking the otehr face detectors
    //cout <<  "Current Face Detector : "<<clm_parameters->curr_face_detector << endl;
    //clm_parameters->curr_face_detector = CLMTracker::CLMParameters::HAAR_DETECTOR; // change to HAAR
    //cout <<  "Current Face Detector : "<<clm_parameters->curr_face_detector << endl;

    //CLMTracker::CLMParameters clm_parameters(arguments);
    CLMTracker::get_video_input_output_params(files, depth_directories, pose_output_files, tracked_videos_output, landmark_output_files, landmark_3D_output_files, use_camera_plane_pose, arguments);
    CLMTracker::get_camera_params(device, fx, fy, cx, cy, arguments);    
    //CLMTracker::CLM clm_model(clm_parameters.model_location);	
    clm_model = new CLMTracker::CLM(clm_parameters->model_location);	
		
    done = false;
    write_settings = false;	
		
    // f_n = -1; 												// for reading from file
    f_n = 0; 													// for reading from device
        
    use_depth = !depth_directories.empty();
    frame_count = 0;
    t1,t0 = cv::getTickCount();									// get the current time as a baseline
    fps = 10;
    visualisation_boundary = 0.2;
        
    cx_undefined = false;
    if(cx == 0 || cy == 0)
      {
	cx_undefined = true;
      }		

		
    ImageOf<PixelRgb> *imgTmp = imageIn.read();  				// read an image
    if (imgTmp != NULL) 
      { 
	//IplImage *cvImage = (IplImage*)imgTmp->getIplImage();
	ImageOf<PixelRgb> &outImage = imageOut.prepare(); 		//get an output image
	outImage.resize(imgTmp->width(), imgTmp->height());		
	outImage = *imgTmp;
	display = (IplImage*) outImage.getIplImage();
	//captured_image = display;
	captured_image = cvarrToMat(display);  // convert to MAT format
      }

	
    // Creating output files
    if(!pose_output_files.empty())
      {
	pose_output_file.open (pose_output_files[f_n], ios_base::out);
      }
	
		
    if(!landmark_output_files.empty())
      {
	landmarks_output_file.open(landmark_output_files[f_n], ios_base::out);
      }

		
    if(!landmark_3D_output_files.empty())
      {
	landmarks_3D_output_file.open(landmark_3D_output_files[f_n], ios_base::out);
      }

    facestate_output_file.open("face_states"+currentDateTime()+".txt", ios_base::out);
    
    //modelControl = false;
    //Frank: read in the matrix for mutual gaze control here
	
    INFO_STREAM( "Starting tracking");

    gazeControl = true;  // true if you want to activate the gaze controller
        
    Property option;
    option.put("device","gazecontrollerclient");
    option.put("remote","/iKinGazeCtrl");
    option.put("local","/client/gaze");

    runGaze = true;   // change to use the gaze controller
    if (runGaze)
      {
	if (!clientGazeCtrl.open(option))
	  {
	    fprintf(stderr,"Error. could not open the gaze controller!");
	    return false;
	  }


	igaze = NULL;
	if (clientGazeCtrl.isValid())
	  {
	    clientGazeCtrl.view(igaze);     // open the view
	  }
	else
	  {
	    INFO_STREAM( "could not open");
	    return false;
	  }

	// latch the controller context in order to preserve it after closing the module
	igaze->storeContext(&startup_context_id);

      }

    gazeChangeFlag = 1;
    oldGazeChangeFlag = gazeChangeFlag;
    oneIter = false;

    /* --- Testing other things
       Property option;
       option.put("device", "remote_controlboard");
       option.put("local", "/test/client");   //local port names
       //option.put("remote", "/icubSim/head"); // for simulation
       option.put("remote", "/icub/head");
    */

    /*
      Property option("(device gazecontrollerclient)");
      option.put("remote","/iKinGazeCtrl");
      option.put("local","/client/gaze"); //("local","/gaze_client");
    */


    /*
      clientGazeCtrl.view(pos);
      clientGazeCtrl.view(vel);
      clientGazeCtrl.view(encs);
    */

    /*
    // set trajectory time:
    igaze->setNeckTrajTime(0.8);
    igaze->setEyesTrajTime(0.4);

    // put the gaze in tracking mode, so that
    // when the torso moves, the gaze controller
    // will compensate for it
    igaze->setTrackingMode(true);
    igaze->bindNeckPitch();
    */

    /*
    // when not using gaze controller
    jnt = 0;

    pos->getAxes(&jnt);
    // 6 joints for the head
    // cout << "----------------------- " << jnt <<"------------------------" << endl;

    encoders.resize(jnt);
    command.resize(jnt);
    tmp.resize(jnt);
    velocity.resize(jnt);
    acceleration.resize(jnt);

    for (int iii = 0; iii < jnt; iii++)
    {
    acceleration[iii] = 50.0;
    }
    pos->setRefAccelerations(acceleration.data());

    for (int iii = 0; iii < jnt; iii++)
    {
    velocity[iii] = 10.0;
    pos->setRefSpeed(iii, velocity[iii]);
    }

    printf("waiting for encoders");
    while(!encs->getEncoders(encoders.data()))
    {
    Time::delay(0.1);
    printf(".");
    }
    printf("\n;");

    command=encoders;
    //now set the shoulder to some value
    command[0]=-10; // up-down (pitch)
    command[1]=0; // roll
    command[2]=-10; // yaw
    command[3]=0;
    command[4]=0;
    command[5]=0;
    pos->positionMove(command.data()); // just to test
    */

    return true;
  }


  // --------------------------------------------------------------
  bool updateModule()
  {

    //cout << __FILE__ << ": " << __LINE__ << endl;
    //cout << "////////////////" << beginTime << "hhhhhhhhhhhhhhhh" << endl;

    ImageOf<PixelRgb> *imgTmp = imageIn.read();  // read an image
    if (imgTmp == NULL) 
      {
	FATAL_STREAM( "Failed to read image!" );
	return true;
      }

    //IplImage *cvImage = (IplImage*)imgTmp->getIplImage();
    ImageOf<PixelRgb> &outImage = imageOut.prepare(); //get an output image
    outImage.resize(imgTmp->width(), imgTmp->height());		
    outImage = *imgTmp;
    display = (IplImage*) outImage.getIplImage();
    //captured_image = display;
    captured_image = cvarrToMat(display); // conver to MAT format

    // If optical centers are not defined just use center of image
    if(cx_undefined)
      {
	cx = captured_image.cols / 2.0f;
	cy = captured_image.rows / 2.0f;
	cx_undefined = true;
      }

    /*
      VideoWriter writerFace;
      if (!write_settings)
      {
      // saving the videos
      if(!tracked_videos_output.empty())
      {
      writerFace = VideoWriter(tracked_videos_output[f_n], CV_FOURCC('D','I','V','X'), 30, captured_image.size(), true);		
      }
      write_settings = true;
      }
    */
    //cout << __FILE__ << ": " << __LINE__ << endl;

    if(captured_image.channels() == 3)
      {
	cvtColor(captured_image, grayscale_image, CV_BGR2GRAY);				
      }
    else
      {
	grayscale_image = captured_image.clone();				
      }
	
    // Get depth image
    if(use_depth)
      {
	char* dst = new char[100];
	std::stringstream sstream;

	sstream << depth_directories[f_n] << "\\depth%05d.png";
	sprintf(dst, sstream.str().c_str(), frame_count + 1);
	// Reading in 16-bit png image representing depth
	depth_image_16_bit = imread(string(dst), -1);

	// Convert to a floating point depth image
	if(!depth_image_16_bit.empty())
	  {
	    depth_image_16_bit.convertTo(depth_image, CV_32F);
	  }
	else
	  {
	    WARN_STREAM( "Can't find depth image" );
	  }
      }

    // The actual facial landmark detection / tracking
    bool detection_success = CLMTracker::DetectLandmarksInVideo(grayscale_image, depth_image, *clm_model, *clm_parameters);


    // Work out the pose of the head from the tracked model
    if(use_camera_plane_pose)
      {
	pose_estimate_CLM = CLMTracker::GetCorrectedPoseCameraPlane(*clm_model, fx, fy, cx, cy, *clm_parameters);
      }
    else
      {
	pose_estimate_CLM = CLMTracker::GetCorrectedPoseCamera(*clm_model, fx, fy, cx, cy, *clm_parameters);
      }

    // Visualising the results
    // Drawing the facial landmarks on the face and the bounding box around it if tracking is successful and initialised
    detection_certainty = clm_model->detection_certainty;

		
    // Only draw if the reliability is reasonable, the value is slightly ad-hoc
    if(detection_certainty < visualisation_boundary)
      {
	//CLMTracker::Draw(captured_image, *clm_model);

	CLMTracker::Draw(captured_image, *clm_model);

	//Frank: this calculation from CLMTracker.cpp is used to 
	//calculate the pose coordinates Tx, Ty, Tz
	//double Z = fx / clm_model.params_global[0];
	//double X = ((clm_model.params_global[4] - cx) * (1.0/fx)) * Z;
	//double Y = ((clm_model.params_global[5] - cy) * (1.0/fy)) * Z;
	//Looks as if I could instead use getShape() in CLM.cpp to get the
	//transformed values of the facial features
	//will be in camera space
	//Will the indicies be consistent?
	//size is 3x68
	// eye features are at 36-48 (6 feature for each eye)
	//left eye 36-41 (viewer's left, not target's left)
	//right eye 42-47
	//eyebrows
	//lips
	//upper lip 48-54 (includes corners)
	//55 - 59 lower outside contour of lip
	//60 - 64 upper inner lip points (includes corners)
	//65-67 lower inner lip points
	//face outline?
	//nose?
	shape_3D = clm_model->GetShape(fx, fy, cx, cy);
	//cout <<  "shape matrix size : "<<shape_3D.rows <<" " << shape_3D.cols << endl; 
	//int ii = 68;
	//double xx = clm_model->detected_landmarks.at<double>(ii);
	//double yy = clm_model->detected_landmarks.at<double>(ii+clm_model->pdm.NumberOfPoints());

	double mean_xx_right_eye = 0;
	double mean_yy_right_eye = 0;
	double mean_x_right_eye3d = 0;
	double mean_y_right_eye3d = 0;
	double mean_z_right_eye3d = 0;
	double mean_xx_left_eye = 0;
	double mean_yy_left_eye = 0;
	double mean_x_left_eye3d = 0;
	double mean_y_left_eye3d = 0;
	double mean_z_left_eye3d = 0;
	double mean_xx_mouth = 0;
	double mean_yy_mouth = 0;
	double mean_x_mouth3d = 0;
	double mean_y_mouth3d = 0;
	double mean_z_mouth3d = 0;

	for (int ii = 36; ii < 48; ii++)
	  {
	    double xx = clm_model->detected_landmarks.at<double>(ii);
	    double yy = clm_model->detected_landmarks.at<double>(ii+clm_model->pdm.NumberOfPoints());
	    
	    double x=shape_3D.at<double>(0,ii);
	    double y=shape_3D.at<double>(1,ii);
	    double z=shape_3D.at<double>(2,ii);

	    //cv::putText(captured_image, "+", cv::Point(xx,yy), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255,255,0));
	    if (ii < 42){
	      //cv::circle(captured_image,cv::Point(xx,yy),3,CV_RGB(255,255,0));
	      mean_xx_right_eye = mean_xx_right_eye + xx;
	      mean_yy_right_eye = mean_yy_right_eye + yy;
	      mean_x_right_eye3d = mean_x_right_eye3d + x;
	      mean_y_right_eye3d = mean_y_right_eye3d + y;
	      mean_z_right_eye3d = mean_z_right_eye3d + z;
	    }
	    else 
	      {
		//cv::circle(captured_image,cv::Point(xx,yy),3,CV_RGB(0,255,0));
		mean_xx_left_eye = mean_xx_left_eye + xx;
		mean_yy_left_eye = mean_yy_left_eye + yy;
	      mean_x_left_eye3d = mean_x_left_eye3d + x;
	      mean_y_left_eye3d = mean_y_left_eye3d + y;
	      mean_z_left_eye3d = mean_z_left_eye3d + z;
	      }
	  }
	mean_xx_left_eye = mean_xx_left_eye / 6;
	mean_yy_left_eye = mean_yy_left_eye / 6;
	mean_x_left_eye3d = mean_x_left_eye3d / 6;
	mean_y_left_eye3d = mean_y_left_eye3d / 6;
	mean_z_left_eye3d = mean_z_left_eye3d / 6;
	pose_left_eye.resize(4);
	pose_left_eye[0] = mean_x_left_eye3d / 1000; //convert to [m]
	pose_left_eye[1] = mean_y_left_eye3d / 1000;
	pose_left_eye[2] = mean_z_left_eye3d / 1000;
	pose_left_eye[3] = 1;

	mean_xx_right_eye = mean_xx_right_eye / 6;
	mean_yy_right_eye = mean_yy_right_eye / 6;
	mean_x_right_eye3d = mean_x_right_eye3d / 6;
	mean_y_right_eye3d = mean_y_right_eye3d / 6;
	mean_z_right_eye3d = mean_z_right_eye3d / 6;
	pose_right_eye.resize(4);
	pose_right_eye[0] = mean_x_right_eye3d / 1000;//convert to [m]
	pose_right_eye[1] = mean_y_right_eye3d / 1000;
	pose_right_eye[2] = mean_z_right_eye3d / 1000;
	pose_right_eye[3] = 1;
	for (int ii = 60; ii < 68; ii++)
	{
	  double xx = clm_model->detected_landmarks.at<double>(ii);
	  double yy = clm_model->detected_landmarks.at<double>(ii+clm_model->pdm.NumberOfPoints());
	  
	  double x=shape_3D.at<double>(0,ii);
	  double y=shape_3D.at<double>(1,ii);
	  double z=shape_3D.at<double>(2,ii);

	  mean_xx_mouth = mean_xx_mouth + xx;
	  mean_yy_mouth = mean_yy_mouth + yy;

	  mean_x_mouth3d = mean_x_mouth3d + x;
	  mean_y_mouth3d = mean_y_mouth3d + y;
	  mean_z_mouth3d = mean_z_mouth3d + z;
	}
	mean_xx_mouth = mean_xx_mouth / 8;
	mean_yy_mouth = mean_yy_mouth / 8;
	mean_x_mouth3d = mean_x_mouth3d / 8;
	mean_y_mouth3d = mean_y_mouth3d / 8;
	mean_z_mouth3d = mean_z_mouth3d / 8;
	pose_mouth.resize(4);
	pose_mouth[0] = mean_x_mouth3d / 1000;//convert to [m]
	pose_mouth[1] = mean_y_mouth3d / 1000;
	pose_mouth[2] = mean_z_mouth3d / 1000;
	pose_mouth[3] = 1;

	//cout << "Mouth 3d position " << mean_x_mouth3d << " " << mean_y_mouth3d << " " << mean_z_mouth3d << endl;
	//cout << "Left eye 3d position " << mean_x_left_eye3d << " " << mean_y_left_eye3d << " " << mean_z_left_eye3d << endl;
	//cout << "Right eye 3d position " << mean_x_right_eye3d << " " << mean_y_right_eye3d << " " << mean_z_right_eye3d << endl;

	//cout << "Pose 3d position " << pose_estimate_CLM[0] << " " << pose_estimate_CLM[1] << " " << pose_estimate_CLM[2] << endl; //" " << pose_estimate_CLM[4] << " " << pose_estimate_CLM[5] << endl;


	Point featurePointLE((int)mean_xx_left_eye, (int)mean_yy_left_eye);
	cv::circle(captured_image, featurePointLE, 1, Scalar(255,255,255), 10.0);
	Point featurePointRE((int)mean_xx_right_eye, (int)mean_yy_right_eye);
	cv::circle(captured_image, featurePointRE, 1, Scalar(255,255,255), 10.0);
	Point featurePointM((int)mean_xx_mouth, (int)mean_yy_mouth);
	cv::circle(captured_image, featurePointM, 1, Scalar(255,255,255), 10.0);

	  //clm_model->pdm.NumberOfPoints() is the total number of points 
	  //used to fit the model
	//for (int i = 0; i < clm_model->pdm.NumberOfPoints() * 3; ++i)
	//  {
	//    landmarks_3D_output_file << " " << shape_3D.at<double>(i);
	//  }

	/*
	// ---------------------------
	// REZA - for Vadim
	// ---------------------------
	//
	// 1- Finding the center of the eyes
	//
	// eye features are at 36-48 (6 feature for each eye)
	double mean_xx_right_eye = 0;
	double mean_yy_right_eye = 0;
	double mean_xx_left_eye = 0;
	double mean_yy_left_eye = 0;


	for (int ii = 36; ii < 48; ii++)
	{
	double xx = clm_model->detected_landmarks.at<double>(ii);
	double yy = clm_model->detected_landmarks.at<double>(ii+clm_model->pdm.NumberOfPoints());


	//cv::putText(captured_image, "+", cv::Point(xx,yy), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255,255,0));
	if (ii < 42){
	//cv::circle(captured_image,cv::Point(xx,yy),3,CV_RGB(255,255,0));
	mean_xx_right_eye = mean_xx_right_eye + xx;
	mean_yy_right_eye = mean_yy_right_eye + yy;
	}
	else 
	{
	//cv::circle(captured_image,cv::Point(xx,yy),3,CV_RGB(0,255,0));
	mean_xx_left_eye = mean_xx_left_eye + xx;
	mean_yy_left_eye = mean_yy_left_eye + yy;
	}
	}
	mean_xx_left_eye = mean_xx_left_eye / 6;
	mean_yy_left_eye = mean_yy_left_eye / 6;
	mean_xx_right_eye = mean_xx_right_eye / 6;
	mean_yy_right_eye = mean_yy_right_eye / 6;


	// 2- Finding the corner of the box

	// initialize with the x,y of the first landmark
	double xx_min = clm_model->detected_landmarks.at<double>(0);
	double xx_max = clm_model->detected_landmarks.at<double>(0);
	double yy_min = clm_model->detected_landmarks.at<double>(clm_model->pdm.NumberOfPoints());
	double yy_max = clm_model->detected_landmarks.at<double>(clm_model->pdm.NumberOfPoints());

	for (int ii=0; ii<clm_model->pdm.NumberOfPoints(); ii++)
	{
	double xx = clm_model->detected_landmarks.at<double>(ii);
	double yy = clm_model->detected_landmarks.at<double>(ii+clm_model->pdm.NumberOfPoints());
	// finding the corners
	if (xx < xx_min)
	xx_min = xx;
	if (xx > xx_max)
	xx_max = xx;
	if (yy < yy_min)
	yy_min = yy;
	if (yy > yy_max);
	yy_max = yy;
	}


	Bottle& poseVadim = targetPort.prepare();
	poseVadim.clear();
	poseVadim.addDouble(xx_min);
	poseVadim.addDouble(yy_min);
	poseVadim.addDouble(xx_max);
	poseVadim.addDouble(yy_max);
	poseVadim.addDouble(mean_xx_left_eye);
	poseVadim.addDouble(mean_yy_left_eye);
	poseVadim.addDouble(mean_xx_right_eye);
	poseVadim.addDouble(mean_yy_right_eye);

	targetPort.write();
	// --------------------------------
	*/

	if(detection_certainty > 1)
	  detection_certainty = 1;
	if(detection_certainty < -1)
	  detection_certainty = -1;

	// cout << "Certainty : " << detection_certainty << "---" << visualisation_boundary << endl;
	detection_certainty = (detection_certainty + 1)/(visualisation_boundary +1);
	//cout << "Normalized Certainty : " << detection_certainty << endl;

	// A rough heuristic for box around the face width
	thickness = (int)std::ceil(2.0* ((double)captured_image.cols) / 640.0);
			
	pose_estimate_to_draw = CLMTracker::GetCorrectedPoseCameraPlane(*clm_model, fx, fy, cx, cy, *clm_parameters);

	// Draw it in reddish if uncertain, blueish if certain
	CLMTracker::DrawBox(captured_image, pose_estimate_to_draw, Scalar((1-detection_certainty)*255.0,0, detection_certainty*255), thickness, fx, fy, cx, cy);

      }

    // Work out the framerate
    if(frame_count % 10 == 0)
      {      
	t1 = cv::getTickCount();
	fps = 10.0 / (double(t1-t0)/cv::getTickFrequency());    // 10.0 because the if is executed every 10 frames and the result has to be averaged over 10
	t0 = t1;
      }
		
    // Write out the framerate on the image before displaying it
    char fpsC[255];
    sprintf(fpsC, "%d", (int)fps);
    string fpsSt("FPS:");
    fpsSt += fpsC;
    cv::putText(captured_image, fpsSt, cv::Point(10,20), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255,0,0));		
    // Output the detected facial landmarks
    /*
      if(!landmark_output_files.empty())
      {
      landmarks_output_file << frame_count + 1 << " " << detection_success;
      for (int i = 0; i < clm_model->pdm.NumberOfPoints() * 2; ++ i)
      {
      landmarks_output_file << " " << clm_model->detected_landmarks.at<double>(i) << " ";
      }
      landmarks_output_file << endl;
      }

      // Output the detected facial landmarks
      if(!landmark_3D_output_files.empty())
      {
      landmarks_3D_output_file << frame_count + 1 << " " << detection_success;
      shape_3D = clm_model->GetShape(fx, fy, cx, cy);
      for (int i = 0; i < clm_model->pdm.NumberOfPoints() * 3; ++i)
      {
      landmarks_3D_output_file << " " << shape_3D.at<double>(i);
      }
      landmarks_3D_output_file << endl;
      }
	double Z = fx / clm_model.params_global[0];
	
		double X = ((clm_model.params_global[4] - cx) * (1.0/fx)) * Z;
		double Y = ((clm_model.params_global[5] - cy) * (1.0/fy)) * Z;
	
      // Output the estimated head pose
      if(!pose_output_files.empty())
      {
      pose_output_file << frame_count + 1 << " " << (float)frame_count * 1000/30 << " " << detection_success << " " << pose_estimate_CLM[0] << " " << pose_estimate_CLM[1] << " " << pose_estimate_CLM[2] << " " << pose_estimate_CLM[3] << " " << pose_estimate_CLM[4] << " " << pose_estimate_CLM[5] << endl;
      }
    */

    //cout << "CLM Estimated pose and quaternion : "  << pose_estimate_CLM[0] << " " << pose_estimate_CLM[1] << " " << pose_estimate_CLM[2] << " " << pose_estimate_CLM[3] << " " << pose_estimate_CLM[4] << " " << pose_estimate_CLM[5] << endl;

    if (runGaze) {
      if ((detection_success) && (abs(pose_estimate_CLM[0]) < 1000 & abs(pose_estimate_CLM[1])< 1000 & abs(pose_estimate_CLM[2]) < 1000 )) {
	//FRANK: This seems to get and compute the pose for individual 
	//features, but this isn't used to generate the poses that
	//the robot looks at (it is done by adding stuff to the head
	//center pose)
	
	//TODO: Move feature calculation from drawing section to here
	
	
	// transforming the pose w.r.t the root of the robot
	igaze->getLeftEyePose(pose_act,ori_act);
	H = axis2dcm(ori_act);
	H(0,3) = pose_act[0];
	H(1,3) = pose_act[1];
	H(2,3) = pose_act[2];
	
	//this should be the center of the face, right?
	pose_clm.resize(4);
	pose_clm[0] = pose_estimate_CLM[0] / 1000; //convert to [m]
	pose_clm[1] = pose_estimate_CLM[1] / 1000;
	pose_clm[2] = pose_estimate_CLM[2] / 1000;
	pose_clm[3] = 1;
	
	
	// ============== Method 2 - using elapsed time =====================
	endTime = clock();
	// start timing

	std::default_random_engine generator(std::random_device{}());
	std::uniform_int_distribution<int> uni_distribution(1,NUM_GAZE_STATES); // use (1,3) for using only eyes and mouth // use (1,5) for 5 points
	std::uniform_real_distribution<double> real_distribution(0,1);

	//here are all the model distribution parameters
	std::discrete_distribution<int> lefteye_distribution({49,852,62,10,1,1,25});
	std::discrete_distribution<int> righteye_distribution({34,31,885,1,7,3,39});
	std::discrete_distribution<int> mouth_distribution({882,38,47,5,6,20,1});
	std::discrete_distribution<int> up_distribution({50,3,8,11,15,905,10});
	std::discrete_distribution<int> down_distribution({3,30,83,9,11,5,860});
	std::discrete_distribution<int> left_distribution({15,20,4,922,4,15,20});
	std::discrete_distribution<int> right_distribution({21,3,34,5,892,20,25});
	std::discrete_distribution<int> rel_distribution({282468,214800,414205,92843,79455,113194,206946});

	//double weights[] = {1, 1, 1,0.5,0.2};
	// cout << "time to keep : " << 3 + timeToKeepAGaze << " diffclock : " << diffclock(endTime,beginTime) << endl;
	//move this to inside the mode switch so that a new state is picked each timestep from the model
	//or recompute the weights to ignore self-transitions
	double currentDiff = abs(diffclock(endTime,beginTime));

	if (gazeMode == model) {
	  if (gazeChangeFlag == none) {
	    gazeChangeFlag = rel_distribution(generator) + 1; 
	  } else {
	    switch(gazeChangeFlag){
	    case lefteye:
	      gazeChangeFlag = lefteye_distribution(generator)+1;
	      break;
	    case righteye:
	      gazeChangeFlag = righteye_distribution(generator)+1;
	      break;
	    case mouth:
	      gazeChangeFlag = mouth_distribution(generator)+1;
	      break;
	    case left:
	      gazeChangeFlag = left_distribution(generator)+1;
	      break;
	    case right:
	      gazeChangeFlag = right_distribution(generator)+1;
	      break;
	    case up:
	      gazeChangeFlag = up_distribution(generator)+1;
	      break;
	    case down:
	      gazeChangeFlag = down_distribution(generator)+1;
	      break;
	    };
	  }

	} else {
	  if ((currentDiff > CYCLING_TIME) || (gazeChangeFlag == none)) {
	    
	    switch(gazeMode){
	    case random:
	      gazeChangeFlag = uni_distribution(generator);
	      break;
	    case fixed:
	      gazeChangeFlag++;
	      if (gazeChangeFlag > NUM_GAZE_STATES)
		gazeChangeFlag = 1;
	      break;
	    };
	  
	    oldGazeChangeFlag = gazeChangeFlag;
	    beginTime = clock();
	    endTime = beginTime; //clock();
	    currentDiff = 0.0;
	  } //if (currentDiff > CYCLING_TIME) 
	} //if (gazemode == model)  

	cout << "################ Changing Gaze >>> mode: " << gazeMode <<  " state: "<< gazeChangeFlag << " timing : " << currentDiff << " looking at : ";
	facestate_output_file << "################ Changing Gaze >>> state: "<< gazeChangeFlag << " timing : " << currentDiff << " looking at : ";
	  
	  switch(gazeChangeFlag){
	  case none:
	    cout << "none" << endl;
	    facestate_output_file << "none" << endl;
	    break;
	  case lefteye:
	    cout << "left eye" << endl;
	    facestate_output_file << "left eye" << endl;
	    break;
	  case righteye:
	    cout << "right eye" << endl;
	    facestate_output_file << "right eye" << endl;
	    break;
	  case mouth:
	    cout << "mouth" << endl;
	    facestate_output_file << "mouth" << endl;
	    break;
	  case left:
	    cout << "left" << endl;
	    facestate_output_file << "left" << endl;
	    break;
	  case right:
	    cout << "right" << endl;
	    facestate_output_file << "right" << endl;
	    break;
	  case up:
	    cout << "up" << endl;
	    facestate_output_file << "up" << endl;
	    break;
	  case down:
	    cout << "down" << endl;
	    facestate_output_file << "down" << endl;
	    break;
	  }
	  
      } else {
	oldGazeChangeFlag = gazeChangeFlag;
	gazeChangeFlag = none;
      } //if detection_success

      fp.resize(4);      
      
      switch(gazeChangeFlag){
      case none:
	//these coordinates taken from the shutdown fixation, look straight ahead
	fp[0]=-1;
	fp[1]=0;
	fp[2]=0.3;
	fp[3]=1;
	//pose_robot = H * fp; //don't need to do the transformation from the left eye, correct?
	pose_robot = fp;
	break;
      case lefteye:
	pose_robot = H * pose_left_eye;
	break;
      case righteye:
	pose_robot = H * pose_right_eye;
	break;
      case mouth:
	pose_robot = H * pose_mouth;
	break;
      case up:
	fp[0] = pose_mouth[0];
	fp[1] = pose_mouth[1] - 0.15;
	fp[2] = pose_mouth[2];
	fp[3] = pose_mouth[3];
	pose_robot = H * fp;
	break;
      case down:
	fp[0] = pose_mouth[0];
	fp[1] = pose_mouth[1] + 0.05;
	fp[2] = pose_mouth[2];
	fp[3] = pose_mouth[3];
	pose_robot = H * fp;
	break;
      case left:
	fp[0] = pose_left_eye[0] + 0.1;
	fp[1] = pose_left_eye[1];
	fp[2] = pose_left_eye[2];
	fp[3] = pose_left_eye[3];
	pose_robot = H * fp;
	break;
      case right:
	fp[0] = pose_right_eye[0] - 0.1;
	fp[1] = pose_right_eye[1];
	fp[2] = pose_right_eye[2];
	fp[3] = pose_right_eye[3];
	pose_robot = H * fp;
	break;
      }
      
      Bottle& fpb = posePort.prepare();
      fpb.clear();
      fpb.addDouble(pose_robot[0]);
      fpb.addDouble(pose_robot[1]);
      fpb.addDouble(pose_robot[2]);
      posePort.write();
      //cout << "pose robot:" << pose_robot[0] << " , " << pose_robot[1] << " , " << pose_robot[2] << endl;
      
    } //if runGaze
    
    //if(!tracked_videos_output.empty())
    //{
    //    writerFace << captured_image;     // output the tracked video
    //}

    frame_count++;                          // Update the frame count
    imageOut.write();

    return true;
  }


  
  // --------------------------------------------------------------
  bool interruptModule()
  {
    cout<<"Interrupting your module, for port cleanup"<<endl;
    //inPort.interrupt();
    imageIn.interrupt();
        
    if (clientGazeCtrl.isValid())
      {

	cout << "restoring the head context ..." << endl;
	fp.resize(3);
	fp[0]=-1;
	fp[1]=0;
	fp[2]=0.3;

	igaze->lookAtFixationPoint(fp); // move the gaze to the desired fixation point
	igaze->waitMotionDone();


	//igaze->restoreContext(startup_context_id); // ... and then retrieve the stored context_0

	//igaze->lookAtFixationPoint(fp);
      }
    return true;
  }
  // --------------------------------------------------------------
  bool close()
  {
    cout<<"Calling close function\n";
    delete clm_parameters;
    delete clm_model;
    //inPort.close();
    //outPort.close();
    targetPort.writeStrict();
    targetPort.close();
    posePort.writeStrict();
    posePort.close();
    imageIn.close();
    imageOut.close();
    clientGazeCtrl.close();
    //frame_count = 0;
    clm_model->Reset();               // reset the model
    pose_output_file.close();
    landmarks_output_file.close();
    facestate_output_file.close();
		
    return true;
  }	
  // --------------------------------------------------------------

protected:

};

// --------------------------------------------------------------
// ----------------
// ----- Main -----
// ----------------
// --------------------------------------------------------------

int main (int argc, char **argv)
{
  //YARP_REGISTER_DEVICES(icubmod);
  Network yarp;

  if (!yarp.checkNetwork())
    {
      fprintf(stderr,"Error. Yarp server not available!");
      return false;
    }



  
  MyModule thisModule;
  ResourceFinder rf;
  cout<<"Object initiated!"<<endl;
  rf.configure(argc,argv);
  thisModule.configure(rf);
  thisModule.runModule();
  return 0;
}

