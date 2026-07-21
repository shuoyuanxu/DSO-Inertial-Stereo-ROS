/**
* ROS1 live wrapper for VI-Stereo-DSO in MONO-INERTIAL mode.
*
* Combines:
*   - dso_ros/src/main.cpp        (live ROS feeding of DSO)
*   - VI-Stereo-DSO main_dso_pangolin.cpp (mono+IMU configuration, run_mono.bash)
*
* Data flow: VI-Stereo-DSO consumes IMU through the global vectors
* m_gry / m_acc / imu_time_stamp and frame timestamps through
* pic_time_stamp[incoming_id] (all declared in util/settings.h).
* This node appends to those vectors from ROS callbacks and calls
* FullSystem::addActiveFrame(img, img, id) once the IMU stream has
* caught up past the frame's timestamp.
*
* IMPORTANT: FullSystem::initFirstFrame_imu() averages the 40 IMU samples
* BEFORE the first frame (m_acc[index-j], j=0..39), so images are dropped
* until at least ~45 IMU samples older than the frame exist.
*
* This file is distributed under GPLv3, like DSO / VI-Stereo-DSO.
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/stat.h>

#include <cmath>
#include <deque>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>

#include <glog/logging.h>

#include "util/settings.h"
#include "util/NumType.h"
#include "util/globalCalib.h"
#include "util/Undistort.h"
#include "util/MinimalImage.h"
#include "util/ImageAndExposure.h"
#include "util/FrameShell.h"
#include "FullSystem/FullSystem.h"
#include "IOWrapper/Output3DWrapper.h"
#include "IOWrapper/Pangolin/PangolinDSOViewer.h"

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointField.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgcodecs.hpp>
#include <polytunnel_vio/SlidingWindowsMsg.h>

#include <map>

using namespace dso;


// Conventional VO visualization over ROS topics:
//   vi_dso/pose         geometry_msgs/PoseStamped  metric camera pose (T_WD-corrected)
//   vi_dso/odom         nav_msgs/Odometry          same pose as odometry
//   vi_dso/path         nav_msgs/Path              full trajectory
//   vi_dso/image_raw    sensor_msgs/Image mono8    undistorted live frame fed to DSO
//   vi_dso/image_points sensor_msgs/Image bgr8     tracking-ref KF with selected points (jet-colored by inverse depth)
//   vi_dso/cloud        sensor_msgs/PointCloud2    sparse map (active+marginalized points of all KFs), world frame
//   vi_dso/sliding_window dso_dense_msgs/SlidingWindowsMsg  one msg per KF in the window, for dense MVS
class ROSOutputWrapper : public IOWrap::Output3DWrapper
{
public:
	explicit ROSOutputWrapper(ros::NodeHandle& nh)
	{
		pubPose  = nh.advertise<geometry_msgs::PoseStamped>("vi_dso/pose", 100);
		pubOdom  = nh.advertise<nav_msgs::Odometry>("vi_dso/odom", 100);
		pubPath  = nh.advertise<nav_msgs::Path>("vi_dso/path", 10);
		pubImg   = nh.advertise<sensor_msgs::Image>("vi_dso/image_raw", 2);
		pubImgPts= nh.advertise<sensor_msgs::Image>("vi_dso/image_points", 2);
		pubCloud = nh.advertise<sensor_msgs::PointCloud2>("vi_dso/cloud", 2);
		pubKfCloud = nh.advertise<sensor_msgs::PointCloud2>("vi_dso/kf_cloud", 20);
		// queue must hold at least a full window, or the MVS node never assembles one
		pubWindow = nh.advertise<polytunnel_vio::SlidingWindowsMsg>("vi_dso/sliding_window", 64);
		path.header.frame_id = "world";
	}

	virtual void publishCamPose(FrameShell* frame, CalibHessian* HCalib) override
	{
		Eigen::Matrix4d m = T_WD.matrix() * frame->camToWorld.matrix() * T_WD.inverse().matrix();
		Eigen::Quaterniond q(Eigen::Matrix3d(m.topLeftCorner<3,3>()));
		q.normalize();

		geometry_msgs::PoseStamped ps;
		if(frame->incoming_id >= 0 && frame->incoming_id < (int)pic_time_stamp.size())
			ps.header.stamp = ros::Time(pic_time_stamp[frame->incoming_id]);
		else
			ps.header.stamp = ros::Time::now();
		ps.header.frame_id = "world";
		ps.pose.position.x = m(0,3);
		ps.pose.position.y = m(1,3);
		ps.pose.position.z = m(2,3);
		ps.pose.orientation.x = q.x();
		ps.pose.orientation.y = q.y();
		ps.pose.orientation.z = q.z();
		ps.pose.orientation.w = q.w();
		pubPose.publish(ps);

		nav_msgs::Odometry od;
		od.header = ps.header;
		od.child_frame_id = "camera";
		od.pose.pose = ps.pose;
		pubOdom.publish(od);

		path.header.stamp = ps.header.stamp;
		path.poses.push_back(ps);
		pubPath.publish(path);

		lastStamp = ps.header.stamp;
		ROS_INFO_THROTTLE(2.0, "T_WD scale: %.3f | path poses: %zu", T_WD.scale(), path.poses.size());
	}

	// undistorted live frame (what DSO actually sees)
	virtual void pushLiveFrame(FrameHessian* fh) override
	{
		if(pubImg.getNumSubscribers() == 0 && !alwaysPublishImages) return;
		int w = wG[0], h = hG[0];
		sensor_msgs::Image img;
		img.header.stamp = lastStamp;
		img.header.frame_id = "camera";
		img.width = w; img.height = h;
		img.encoding = "mono8";
		img.step = w;
		img.data.resize(w*h);
		for(int i = 0; i < w*h; i++)
		{
			float v = fh->dI[i][0];
			img.data[i] = (unsigned char)(v < 0 ? 0 : (v > 255 ? 255 : v));
		}
		pubImg.publish(img);
	}

	// tracking-reference keyframe overlaid with selected points (jet color = inverse depth)
	virtual void pushDepthImage(MinimalImageB3* image) override
	{
		if(pubImgPts.getNumSubscribers() == 0 && !alwaysPublishImages) return;
		sensor_msgs::Image img;
		img.header.stamp = lastStamp;
		img.header.frame_id = "camera";
		img.width = image->w; img.height = image->h;
		img.encoding = "bgr8";
		img.step = image->w * 3;
		img.data.resize(image->w * image->h * 3);
		memcpy(img.data.data(), image->data, image->w * image->h * 3);
		pubImgPts.publish(img);
	}

	// sparse point cloud: accumulate per-keyframe points (camera frame + pose),
	// rebuild in world frame with the CURRENT scale/gravity alignment each time
	virtual void publishKeyframes(std::vector<FrameHessian*>& frames, bool final, CalibHessian* HCalib) override
	{
		if(final) return;
		float fxi = 1.0f/HCalib->fxl(), fyi = 1.0f/HCalib->fyl();
		float cxi = -HCalib->cxl()/HCalib->fxl(), cyi = -HCalib->cyl()/HCalib->fyl();

		for(FrameHessian* fh : frames)
		{
			KFCloud& kc = clouds[fh->frameID];
			kc.camToWorld = fh->shell->camToWorld;
			kc.stamp = (fh->shell->incoming_id >= 0 && fh->shell->incoming_id < (int)pic_time_stamp.size())
			           ? pic_time_stamp[fh->shell->incoming_id] : 0;
			kc.pts.clear();
			auto addPts = [&](const std::vector<PointHessian*>& v)
			{
				for(PointHessian* p : v)
				{
					if(p == 0 || !std::isfinite(p->idepth_scaled) || p->idepth_scaled <= 0) continue;
					float d = 1.0f / p->idepth_scaled;
					if(d > 100.0f) continue;   // drop near-infinity points, they wreck the RViz view
					kc.pts.push_back(Eigen::Vector3f((p->u*fxi + cxi)*d, (p->v*fyi + cyi)*d, d));
				}
			};
			addPts(fh->pointHessians);
			addPts(fh->pointHessiansMarginalized);

			// per-keyframe camera-frame cloud for downstream fusion (dso_imu_graph
			// re-anchors these with its own corrected poses)
			if(pubKfCloud.getNumSubscribers() > 0 && kc.stamp > 0 && !kc.pts.empty())
			{
				sensor_msgs::PointCloud2 pc;
				pc.header.stamp = ros::Time(kc.stamp);
				pc.header.frame_id = "camera";
				pc.height = 1;
				pc.fields.resize(3);
				const char* nm[3] = {"x","y","z"};
				for(int i = 0; i < 3; i++)
				{
					pc.fields[i].name = nm[i];
					pc.fields[i].offset = 4*i;
					pc.fields[i].datatype = sensor_msgs::PointField::FLOAT32;
					pc.fields[i].count = 1;
				}
				pc.point_step = 12;
				pc.is_bigendian = false;
				pc.is_dense = true;
				pc.width = kc.pts.size();
				pc.row_step = 12 * pc.width;
				pc.data.resize(pc.row_step);
				memcpy(pc.data.data(), kc.pts.data(), pc.row_step);
				pubKfCloud.publish(pc);
			}
		}

		publishSlidingWindow(frames, HCalib);

		if(pubCloud.getNumSubscribers() == 0 && !alwaysPublishImages) return;

		sensor_msgs::PointCloud2 pc;
		pc.header.stamp = lastStamp;
		pc.header.frame_id = "world";
		pc.height = 1;
		pc.fields.resize(3);
		const char* names[3] = {"x","y","z"};
		for(int i = 0; i < 3; i++)
		{
			pc.fields[i].name = names[i];
			pc.fields[i].offset = 4*i;
			pc.fields[i].datatype = sensor_msgs::PointField::FLOAT32;
			pc.fields[i].count = 1;
		}
		pc.point_step = 12;
		pc.is_bigendian = false;
		pc.is_dense = true;

		size_t n = 0;
		for(auto& kv : clouds) n += kv.second.pts.size();
		pc.width = n;
		pc.row_step = pc.point_step * n;
		pc.data.resize(pc.row_step);
		float* out = reinterpret_cast<float*>(pc.data.data());
		for(auto& kv : clouds)
		{
			for(const Eigen::Vector3f& p : kv.second.pts)
			{
				// dso-world -> metric world via the current Sim3 alignment
				Eigen::Vector3d pw = T_WD * (kv.second.camToWorld * p.cast<double>());
				*out++ = (float)pw.x(); *out++ = (float)pw.y(); *out++ = (float)pw.z();
			}
		}
		pubCloud.publish(pc);
	}

	// One SlidingWindowsMsg per keyframe in the current window, all sharing msg_id.
	// This is what the dense (MVS / stereo) depth node consumes: it needs the image
	// DSO actually optimized together with that frame's own pose, which neither
	// pushLiveFrame (latest frame, not a KF) nor pubKfCloud (points, no image) gives.
	void publishSlidingWindow(std::vector<FrameHessian*>& frames, CalibHessian* HCalib)
	{
		if(pubWindow.getNumSubscribers() == 0) return;
		if(frames.empty() || frames.size() > 255) return;

		// Sim3 world alignment, split so the pose stays a *rigid* transform:
		// an MVS plane sweep needs an orthonormal rotation block, so the scale
		// goes into the translation only. Depths then come out already metric.
		const double s = T_WD.scale();
		if(!std::isfinite(s) || s <= 0) return;
		const Eigen::Matrix3d R_WD = T_WD.rxso3().matrix() / s;
		const Eigen::Vector3d t_WD = T_WD.translation();

		// Plane-sweep depth range from DSO's own sparse depths over the whole
		// window (metric). The reference implementation hardcoded 0.01-10 m,
		// which is wrong for any rig but the one it was tuned on.
		std::vector<float> depths;
		for(FrameHessian* fh : frames)
		{
			auto it = clouds.find(fh->frameID);
			if(it == clouds.end()) continue;
			for(const Eigen::Vector3f& p : it->second.pts)
				if(p.z() > 0 && std::isfinite(p.z())) depths.push_back(p.z() * (float)s);
		}
		if(depths.size() < 32) return;   // too few points to trust a range yet
		std::sort(depths.begin(), depths.end());
		float dmin = depths[(size_t)(0.05 * (depths.size()-1))];
		float dmax = depths[(size_t)(0.95 * (depths.size()-1))];
		// widen past the sparse envelope: MVS should be able to find structure
		// DSO's point selector skipped, and a too-tight range clips it silently
		dmin *= 0.7f;
		dmax *= 1.4f;
		if(!std::isfinite(dmin) || !std::isfinite(dmax) || dmax <= dmin) return;
		if(dmin < 0.05f) dmin = 0.05f;

		// count usable views FIRST: window_size must equal the number actually
		// published, or the subscriber blocks forever waiting for a frame whose
		// image was already freed
		size_t nValid = 0;
		for(FrameHessian* fh : frames) if(fh->dI != 0) nValid++;
		if(nValid < 2) return;   // a plane sweep needs a reference plus a source

		const int w = wG[0], h = hG[0];
		const uint64_t id = ++windowId;
		uint8_t idx = 0;
		for(FrameHessian* fh : frames)
		{
			if(fh->dI == 0) continue;   // image freed; can't use it as an MVS view

			polytunnel_vio::SlidingWindowsMsg m;
			m.msg_id = id;
			m.window_size = (uint8_t)nValid;
			m.index = idx++;
			m.depth_min = dmin;
			m.depth_max = dmax;

			auto it = clouds.find(fh->frameID);
			double stamp = (it != clouds.end()) ? it->second.stamp : 0;
			m.image.header.stamp = stamp > 0 ? ros::Time(stamp) : lastStamp;
			m.image.header.frame_id = "camera";
			m.image.width = w; m.image.height = h;
			m.image.encoding = "mono8";       // cv_bridge widens to bgr8 downstream
			m.image.step = w;
			m.image.data.resize(w*h);
			for(int i = 0; i < w*h; i++)
			{
				float v = fh->dI[i][0];
				m.image.data[i] = (unsigned char)(v < 0 ? 0 : (v > 255 ? 255 : v));
			}

			// rigid metric cam->world, row-major, consistent with vi_dso/cloud:
			// p_world = R * p_cam + t
			const SE3& cw = fh->shell->camToWorld;
			Eigen::Matrix3d R = R_WD * cw.rotationMatrix();
			Eigen::Vector3d t = s * (R_WD * cw.translation()) + t_WD;
			for(int r = 0; r < 3; r++)
			{
				for(int c = 0; c < 3; c++) m.camToWorld[4*r + c] = R(r,c);
				m.camToWorld[4*r + 3] = t(r);
			}
			m.camToWorld[12] = 0; m.camToWorld[13] = 0;
			m.camToWorld[14] = 0; m.camToWorld[15] = 1;

			// level-0 intrinsics, already in pixels of the published image
			m.Intrinsics[0] = HCalib->fxl();
			m.Intrinsics[1] = HCalib->fyl();
			m.Intrinsics[2] = HCalib->cxl();
			m.Intrinsics[3] = HCalib->cyl();

			pubWindow.publish(m);
		}
		ROS_INFO_THROTTLE(5.0, "sliding_window %lu: %u KFs, depth %.2f-%.2f m (scale %.3f)",
		                  (unsigned long)id, idx, dmin, dmax, s);
	}

	virtual void reset() override
	{
		path.poses.clear();
		clouds.clear();
	}

	bool alwaysPublishImages = false;

private:
	struct KFCloud
	{
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		SE3 camToWorld;
		double stamp = 0;
		std::vector<Eigen::Vector3f> pts;
	};
	// SE3 members need 32-byte alignment under -march=native: use Eigen's allocator
	std::map<int, KFCloud, std::less<int>,
	         Eigen::aligned_allocator<std::pair<const int, KFCloud>>> clouds;
	ros::Publisher pubPose, pubOdom, pubPath, pubImg, pubImgPts, pubCloud, pubKfCloud;
	ros::Publisher pubWindow;
	uint64_t windowId = 0;
	nav_msgs::Path path;
	ros::Time lastStamp;
};


FullSystem* fullSystem = 0;
Undistort* undistorter = 0;
Undistort* undistorterR = 0;      // stereo mode only

std::deque<std::pair<double, cv::Mat>> imageQueue;
std::deque<std::pair<double, cv::Mat>> imageQueueR;   // raw right images (stereo mode)
bool linearizeOp = true;   // true = synchronous mapping (the code path VI-DSO's authors tested);
                           // false = async mapping thread (patched, but less battle-tested)
bool useImu = true;        // mono_imu / stereo_imu vs pure vision
bool useStereoMode = false;
double stereoSyncTol = 0.008;     // max |t_left - t_right| to pair frames
double lastImuTime = -1;
double lastFedImageTime = -1;
double imuCatchupMargin = 0.01;   // require IMU this far past the frame time
double timeshift_cam_imu = 0.0;   // Kalibr: t_imu = t_cam + timeshift
bool vioStarted = false;
size_t maxImageQueue = 40;


// Same file format as VI-Stereo-DSO getIMUinfo():
// rows 0-2 of T_BC (imu <- cam, 3x4), a skipped 4th row, then
// gyr_noise, acc_noise, gyr_walk, acc_walk (one per line).
void loadImuInfo(const std::string& path)
{
	std::ifstream inf(path.c_str());
	if(!inf.good())
	{
		ROS_FATAL("cannot open imu_info file %s", path.c_str());
		exit(1);
	}
	std::string sline;
	int line = 0;
	Mat33 R;
	Vec3 t;
	Vec4 noise;
	while(line < 3 && std::getline(inf, sline))
	{
		std::istringstream ss(sline);
		for(int i = 0; i < 3; ++i) ss >> R(line, i);
		ss >> t(line);
		++line;
	}
	std::getline(inf, sline);
	++line;
	while(line < 8 && std::getline(inf, sline))
	{
		std::istringstream ss(sline);
		ss >> noise(line - 4);
		++line;
	}
	inf.close();

	T_BC = SE3(R, t);
	GyrCov = Mat33::Identity() * noise(0) * noise(0) / 0.005;
	AccCov = Mat33::Identity() * noise(1) * noise(1) / 0.005;
	GyrRandomWalkNoise = Mat33::Identity() * noise(2) * noise(2);
	AccRandomWalkNoise = Mat33::Identity() * noise(3) * noise(3);

	ROS_INFO_STREAM("T_BC:\n" << T_BC.matrix());
	ROS_INFO_STREAM("IMU noise (gyr_n acc_n gyr_w acc_w): " << noise.transpose());
}


void resetFullSystem()
{
	std::vector<IOWrap::Output3DWrapper*> wraps = fullSystem->outputWrapper;
	delete fullSystem;
	for(IOWrap::Output3DWrapper* ow : wraps) ow->reset();

	fullSystem = new FullSystem();
	fullSystem->linearizeOperation = linearizeOp;
	fullSystem->outputWrapper = wraps;
	if(undistorter->photometricUndist != 0)
		fullSystem->setGammaFunction(undistorter->photometricUndist->getG());

	setting_fullResetRequested = false;
	first_track_flag = false;   // as in main_dso_pangolin.cpp reset path
}


void feedFrame(double time, const cv::Mat& gray, const cv::Mat& grayRight = cv::Mat())
{
	int id = (int)pic_time_stamp.size();
	pic_time_stamp.push_back(time);

	MinimalImageB minImg((int)gray.cols, (int)gray.rows, (unsigned char*)gray.data);
	ImageAndExposure* undistImg = undistorter->undistort<unsigned char>(&minImg, 1, time, 1.0f);
	ImageAndExposure* undistImgR = 0;
	if(useStereoMode && !grayRight.empty())
	{
		MinimalImageB minImgR((int)grayRight.cols, (int)grayRight.rows, (unsigned char*)grayRight.data);
		undistImgR = undistorterR->undistort<unsigned char>(&minImgR, 1, time, 1.0f);
	}
	fullSystem->addActiveFrame(undistImg, undistImgR ? undistImgR : undistImg, id);   // mono: left == right
	delete undistImg;
	if(undistImgR) delete undistImgR;

	if(fullSystem->initFailed || setting_fullResetRequested)
	{
		ROS_WARN("VI-DSO: init failed / reset requested -> resetting full system");
		resetFullSystem();
	}
	else if(fullSystem->isLost)
	{
		ROS_WARN("VI-DSO: tracking LOST -> resetting full system");
		resetFullSystem();
	}
}


void processImageQueue()
{
	while(!imageQueue.empty())
	{
		double t = imageQueue.front().first;

		if(useImu)
		{
			if(!vioStarted)
			{
				// initFirstFrame_imu needs >= 40 IMU samples strictly before
				// the first frame; keep a small safety margin.
				if(imu_time_stamp.size() < 45) return;
				if(t <= imu_time_stamp[44])
				{
					imageQueue.pop_front();   // too early, no IMU history yet
					continue;
				}
				vioStarted = true;
			}

			if(lastImuTime < t + imuCatchupMargin) return;   // wait for IMU to catch up
		}

		cv::Mat right;
		if(useStereoMode)
		{
			// pair with the closest right image; drop stale rights
			while(!imageQueueR.empty() && imageQueueR.front().first < t - stereoSyncTol)
				imageQueueR.pop_front();
			if(imageQueueR.empty()) return;              // right not arrived yet
			if(imageQueueR.front().first > t + stereoSyncTol)
			{
				imageQueue.pop_front();                  // no partner for this left frame
				continue;
			}
			right = imageQueueR.front().second;
		}

		feedFrame(t, imageQueue.front().second, right);
		imageQueue.pop_front();
	}
}


void imuCb(const sensor_msgs::ImuConstPtr& msg)
{
	double t = msg->header.stamp.toSec();
	if(t <= lastImuTime)
	{
		ROS_WARN_THROTTLE(5.0, "dropping out-of-order IMU message");
		return;
	}
	Vec3 gyr(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
	Vec3 acc(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
	m_gry.push_back(gyr);
	m_acc.push_back(acc);
	imu_time_stamp.push_back(t);
	lastImuTime = t;

	processImageQueue();
}


void handleImage(double stamp, const cv::Mat& gray)
{
	double t = stamp + timeshift_cam_imu;   // camera time -> IMU clock
	if(t <= lastFedImageTime) return;
	lastFedImageTime = t;

	imageQueue.push_back(std::make_pair(t, gray));
	if(imageQueue.size() > maxImageQueue)
	{
		ROS_WARN_THROTTLE(5.0, "VI-DSO cannot keep up, dropping oldest queued frame");
		imageQueue.pop_front();
	}
	processImageQueue();
}


void imgCompressedCb(const sensor_msgs::CompressedImageConstPtr& msg)
{
	cv::Mat gray = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_GRAYSCALE);
	if(gray.empty())
	{
		ROS_WARN("failed to decode compressed image");
		return;
	}
	handleImage(msg->header.stamp.toSec(), gray);
}


void imgRightCompressedCb(const sensor_msgs::CompressedImageConstPtr& msg)
{
	cv::Mat gray = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_GRAYSCALE);
	if(gray.empty()) return;
	imageQueueR.push_back(std::make_pair(msg->header.stamp.toSec() + timeshift_cam_imu, gray));
	while(imageQueueR.size() > 2*maxImageQueue) imageQueueR.pop_front();
	processImageQueue();
}


// Same format as VI-Stereo-DSO getTstereo(): 3 lines of R|t for T_C0C1 (left <- right)
void loadTStereo(const std::string& path)
{
	std::ifstream inf(path.c_str());
	if(!inf.good())
	{
		ROS_FATAL("cannot open t_stereo file %s", path.c_str());
		exit(1);
	}
	std::string sline;
	int line = 0;
	Mat33 R;
	Vec3 t;
	while(line < 3 && std::getline(inf, sline))
	{
		std::istringstream ss(sline);
		for(int i = 0; i < 3; ++i) ss >> R(line, i);
		ss >> t(line);
		++line;
	}
	inf.close();
	SE3 temp(R, t);
	T_C0C1 = temp;
	T_C1C0 = temp.inverse();
	ROS_INFO_STREAM("T_C0C1:\n" << T_C0C1.matrix());
}


void imgRawCb(const sensor_msgs::ImageConstPtr& msg)
{
	cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO8);
	handleImage(msg->header.stamp.toSec(), cv_ptr->image);
}


int main(int argc, char** argv)
{
	ros::init(argc, argv, "vi_dso_live");
	ros::NodeHandle nh;
	ros::NodeHandle pnh("~");

	google::InitGoogleLogging(argv[0]);

	std::string calibFile, imuInfoFile, gammaFile, vignetteFile;
	std::string calibRightFile, tStereoFile, imageTopicRight;
	std::string imageTopic, imuTopic;
	bool useCompressed, nogui, useMultiThreading, carryState;
	double gNorm, imuWeight, imuWeightTracker, reserveHours;

	pnh.param<std::string>("calib", calibFile, "");
	pnh.param<std::string>("imu_info", imuInfoFile, "");
	pnh.param<std::string>("gamma", gammaFile, "");
	pnh.param<std::string>("vignette", vignetteFile, "");
	pnh.param<std::string>("image_topic", imageTopic, "/forwardLeft/image_raw/compressed");
	pnh.param<std::string>("imu_topic", imuTopic, "/ms/imu/data");
	pnh.param<bool>("use_compressed", useCompressed, true);

	// mode selection: mono (F,F), mono-inertial (T,F), stereo (F,T), stereo-inertial (T,T)
	pnh.param<bool>("use_imu", useImu, true);
	pnh.param<bool>("use_stereo", useStereoMode, false);
	pnh.param<bool>("carry_state", carryState, true);   // keep bias/scale across re-inits
	pnh.param<std::string>("calib_right", calibRightFile, "");
	pnh.param<std::string>("t_stereo", tStereoFile, "");
	pnh.param<std::string>("image_topic_right", imageTopicRight, "/forwardRight/image_raw/compressed");
	pnh.param<double>("stereo_sync_tol", stereoSyncTol, 0.008);
	pnh.param<bool>("nogui", nogui, false);
	pnh.param<bool>("multithreading", useMultiThreading, false);   // VI-DSO's MT path is untested upstream (all their scripts use nomt=1)
	pnh.param<bool>("linearize_operation", linearizeOp, true);
	pnh.param<double>("timeshift_cam_imu", timeshift_cam_imu, 0.0);
	pnh.param<double>("imu_catchup_margin", imuCatchupMargin, 0.01);
	pnh.param<double>("g_norm", gNorm, 9.81);
	pnh.param<double>("imu_weight", imuWeight, 6.0);
	pnh.param<double>("imu_weight_tracker", imuWeightTracker, 0.6);
	pnh.param<double>("reserve_hours", reserveHours, 2.0);

	// OKVIS-style relaxed initialization: accept early, let scale converge online
	double initSlack, scaleLow, scaleHigh, baHessian, bgHessian;
	pnh.param<double>("init_slack", initSlack, 3.0);          // multiplies the KF2-4 rmse gates.
	                                                         // 3.0: A/B-verified clean single init on
	                                                         // low-excitation data; 1.0 causes reset storms
	pnh.param<double>("scale_reset_low", scaleLow, 0.1);      // T_WD scale bounds before full reset
	pnh.param<double>("scale_reset_high", scaleHigh, 10.0);
	// bias priors (OKVIS keeps a persistent accel-bias prior so bias can't absorb
	// the gravity/scale ambiguity during constant-velocity motion; run_mono used 0)
	pnh.param<double>("init_ba_hessian", baHessian, 0.0);
	pnh.param<double>("init_bg_hessian", bgHessian, 0.0);
	double scaleHessian;
	pnh.param<double>("init_scale_hessian", scaleHessian, 0.0);  // prior on the (weakly observable) scale state
	double stereoMaxDepth;
	pnh.param<double>("stereo_max_depth", stereoMaxDepth, 1e10);  // range-gate static stereo (m); off by default
	benchmark_initializerSlackFactor = (float)initSlack;
	setting_scaleResetLow = scaleLow;
	setting_scaleResetHigh = scaleHigh;
	setting_stereoMaxDepth = stereoMaxDepth;

	if(calibFile.empty())
	{
		ROS_FATAL("param ~calib is required");
		return 1;
	}
	if(useImu && imuInfoFile.empty())
	{
		ROS_FATAL("param ~imu_info is required when use_imu is true");
		return 1;
	}
	if(useStereoMode && (calibRightFile.empty() || tStereoFile.empty()))
	{
		ROS_FATAL("params ~calib_right and ~t_stereo are required when use_stereo is true");
		return 1;
	}

	// ===== settings: preset=0 (from settingsDefault) =====
	setting_desiredImmatureDensity = 1500;
	setting_desiredPointDensity = 2000;
	setting_minFrames = 5;
	setting_maxFrames = 7;
	setting_maxOptIterations = 6;
	setting_minOptIterations = 1;
	setting_logStuff = false;
	setting_kfGlobalWeight = 1;
	setting_maxShiftWeightT = 0.04f * (640 + 128);
	setting_maxShiftWeightR = 0.04f * (640 + 128);
	setting_maxShiftWeightRT = 0.02f * (640 + 128);

	// ===== mode=1: photometric mode without gamma calibration =====
	setting_photometricCalibration = 0;
	setting_affineOptModeA = 0;
	setting_affineOptModeB = 0;

	// ===== visual(-inertial) config (main_dso_pangolin.cpp + run scripts) =====
	imu_weight = imuWeight;
	imu_weight_tracker = imuWeightTracker;
	stereo_weight = useStereoMode ? 2.0 : 0;
	use_stereo = useStereoMode;
	G_norm = gNorm;
	imu_use_flag = useImu;
	imu_track_flag = useImu;
	setting_carryStateOverReset = carryState && useImu;
	ROS_INFO("VI-DSO mode: %s%s | carry_state: %s",
			 useStereoMode ? "stereo" : "mono", useImu ? "+imu" : " (vision only)",
			 (carryState && useImu) ? "on" : "off");
	use_optimize = true;
	imu_track_ready = false;
	use_Dmargin = true;
	setting_initialIMUHessian = 0;
	setting_initialScaleHessian = (float)scaleHessian;
	setting_initialbaHessian = (float)baHessian;
	setting_initialbgHessian = (float)bgHessian;
	imu_lambda = 5;
	d_min = sqrt(1.1);
	setting_margWeightFac_imu = 0.25;
	multiThreading = useMultiThreading;
	savefile_tail = "vi_dso_ros";
	disableAllDisplay = nogui;

	// The mapping thread reads the IMU vectors while callbacks append to
	// them; reserving up-front prevents reallocation underneath a reader.
	size_t nImu = (size_t)(reserveHours * 3600.0 * 250.0);
	size_t nPic = (size_t)(reserveHours * 3600.0 * 35.0);
	m_gry.reserve(nImu);
	m_acc.reserve(nImu);
	imu_time_stamp.reserve(nImu);
	pic_time_stamp.reserve(nPic);

	if(useImu) loadImuInfo(imuInfoFile);

	mkdir("./data", 0755);   // savetrajectory_tum appends to ./data/<savefile_tail>.txt

	undistorter = Undistort::getUndistorterForFile(calibFile, gammaFile, vignetteFile);
	setGlobalCalib(
			(int)undistorter->getSize()[0],
			(int)undistorter->getSize()[1],
			undistorter->getK().cast<float>());

	if(useStereoMode)
	{
		undistorterR = Undistort::getUndistorterForFile(calibRightFile, "", "");
		if((int)undistorterR->getSize()[0] != (int)undistorter->getSize()[0] ||
		   (int)undistorterR->getSize()[1] != (int)undistorter->getSize()[1])
		{
			ROS_FATAL("left/right undistorted output sizes differ");
			return 1;
		}
		K_right = undistorterR->getK().cast<float>();
		loadTStereo(tStereoFile);
		ROS_INFO("stereo baseline (from calib_right line 5): %.4f m", baseline);
	}

	fullSystem = new FullSystem();
	fullSystem->linearizeOperation = linearizeOp;

	if(!nogui)
		fullSystem->outputWrapper.push_back(new IOWrap::PangolinDSOViewer(
				(int)undistorter->getSize()[0],
				(int)undistorter->getSize()[1]));

	fullSystem->outputWrapper.push_back(new ROSOutputWrapper(nh));

	if(undistorter->photometricUndist != 0)
		fullSystem->setGammaFunction(undistorter->photometricUndist->getG());

	ros::Subscriber imuSub;
	if(useImu) imuSub = nh.subscribe(imuTopic, 2000, &imuCb);
	ros::Subscriber imgSubC, imgSubR, imgSubRight;
	if(useCompressed)
		imgSubC = nh.subscribe(imageTopic, 5, &imgCompressedCb);
	else
		imgSubR = nh.subscribe(imageTopic, 5, &imgRawCb);
	if(useStereoMode)
		imgSubRight = nh.subscribe(imageTopicRight, 5, &imgRightCompressedCb);

	ROS_INFO("vi_dso_live ready. image: %s (%s)%s%s%s",
			 imageTopic.c_str(), useCompressed ? "compressed" : "raw",
			 useStereoMode ? ", right: " : "", useStereoMode ? imageTopicRight.c_str() : "",
			 useImu ? ", imu on" : ", imu off");

	ros::spin();   // single-threaded spinner: image & IMU callbacks are serialized

	fullSystem->blockUntilMappingIsFinished();
	fullSystem->printResult("result.txt");

	for(IOWrap::Output3DWrapper* ow : fullSystem->outputWrapper)
	{
		ow->join();
		delete ow;
	}
	delete fullSystem;
	delete undistorter;
	return 0;
}
