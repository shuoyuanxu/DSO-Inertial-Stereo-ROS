/**
 * Fuse per-keyframe dense depth into a global point cloud.
 *
 * Subscribes  dense_depth/depth_info  (dso_dense_msgs/DepthMsg)
 * Publishes   dense_mapping/cloud     (sensor_msgs/PointCloud2, frame "world")
 *             dense_mapping/kf_cloud  (latest keyframe only, for a live view)
 *
 * Backend-agnostic: DepthMsg carries the metric depth, the confidence and the
 * rigid cam->world of its reference keyframe, so UniMVSNet and stereo fuse
 * through exactly the same path.
 *
 * Differences from the reference implementation this is modelled on:
 *   - the global map is voxel-filtered incrementally and capped, instead of
 *     growing without bound until the process is killed;
 *   - fusion runs on a worker thread and publishes on a timer, so a slow
 *     filter pass never blocks the subscriber callback;
 *   - points are gated on confidence AND on a depth range, since a plane
 *     sweep will happily return confident values at its range limits.
 *
 * GPLv3, consistent with the rest of this workspace.
 */
#include <mutex>
#include <thread>
#include <condition_variable>
#include <deque>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Dense>

#include <polytunnel_vio/DepthMsg.h>

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> CloudT;

class DenseMapper
{
public:
	DenseMapper(ros::NodeHandle& nh, ros::NodeHandle& pnh)
	{
		pnh.param("resolution",     resolution_,     0.05);
		pnh.param("conf_threshold", confThreshold_,  0.5);
		pnh.param("min_depth",      minDepth_,       0.3);
		pnh.param("max_depth",      maxDepth_,      15.0);
		pnh.param("pixel_step",     pixelStep_,      2);
		pnh.param("max_points",     maxPoints_,      4000000);
		pnh.param("publish_period", publishPeriod_,  1.0);

		global_ = CloudT::Ptr(new CloudT);
		voxel_.setLeafSize(resolution_, resolution_, resolution_);

		pubCloud_   = nh.advertise<sensor_msgs::PointCloud2>("dense_mapping/cloud", 1, true);
		pubKfCloud_ = nh.advertise<sensor_msgs::PointCloud2>("dense_mapping/kf_cloud", 2);
		sub_ = nh.subscribe("dense_depth/depth_info", 20, &DenseMapper::onDepth, this);

		worker_ = std::thread(&DenseMapper::loop, this);
		timer_  = nh.createTimer(ros::Duration(publishPeriod_), &DenseMapper::onTimer, this);

		ROS_INFO("dense_mapping: voxel %.3f m, conf>%.2f, depth %.1f-%.1f m, "
		         "step %d, cap %d pts",
		         resolution_, confThreshold_, minDepth_, maxDepth_,
		         pixelStep_, maxPoints_);
	}

	~DenseMapper()
	{
		{
			std::unique_lock<std::mutex> lk(mtx_);
			stop_ = true;
			cv_.notify_all();
		}
		if(worker_.joinable()) worker_.join();
	}

private:
	void onDepth(const polytunnel_vio::DepthMsg::ConstPtr& msg)
	{
		std::unique_lock<std::mutex> lk(mtx_);
		// bounded: under load drop the oldest, keep fusing the freshest geometry
		if(queue_.size() > 8) { queue_.pop_front(); ++nDropped_; }
		queue_.push_back(msg);
		cv_.notify_one();
	}

	void loop()
	{
		while(ros::ok())
		{
			polytunnel_vio::DepthMsg::ConstPtr msg;
			{
				std::unique_lock<std::mutex> lk(mtx_);
				cv_.wait(lk, [&]{ return stop_ || !queue_.empty(); });
				if(stop_) return;
				msg = queue_.front();
				queue_.pop_front();
			}
			try { fuse(msg); }
			catch(const std::exception& e)
			{ ROS_ERROR("dense_mapping fuse failed: %s", e.what()); }
		}
	}

	void fuse(const polytunnel_vio::DepthMsg::ConstPtr& msg)
	{
		cv_bridge::CvImageConstPtr d = cv_bridge::toCvShare(msg->depth, msg, "32FC1");
		cv_bridge::CvImageConstPtr c = cv_bridge::toCvShare(msg->confidence, msg, "32FC1");
		// bgr8, not mono8: dense_depth resamples raw colour into the keyframe
		// grid, and asking cv_bridge for mono8 would throw that away again.
		// A grayscale keyframe (colorize off, or no colour frame in time)
		// widens to bgr8 here, so this path handles both.
		cv_bridge::CvImagePtr      g = cv_bridge::toCvCopy(msg->image, "bgr8");
		if(d->image.empty() || c->image.empty()) return;

		const double fx = msg->Intrinsics[0], fy = msg->Intrinsics[1];
		const double cx = msg->Intrinsics[2], cy = msg->Intrinsics[3];
		if(!(fx > 0 && fy > 0)) { ROS_WARN_THROTTLE(5, "bad intrinsics"); return; }

		// row-major cam->world, exactly as vi_dso_live packed it
		Eigen::Matrix4d T;
		for(int r = 0; r < 4; r++)
			for(int col = 0; col < 4; col++)
				T(r, col) = msg->camToWorld[4*r + col];
		if(!T.allFinite()) { ROS_WARN_THROTTLE(5, "non-finite pose"); return; }

		const Eigen::Matrix3d R = T.topLeftCorner<3,3>();
		const Eigen::Vector3d t = T.topRightCorner<3,1>();

		CloudT::Ptr kf(new CloudT);
		kf->reserve((d->image.rows / pixelStep_) * (d->image.cols / pixelStep_));

		const bool haveColor = (g->image.rows == d->image.rows &&
		                        g->image.cols == d->image.cols);

		for(int y = 0; y < d->image.rows; y += pixelStep_)
		{
			const float* dr = d->image.ptr<float>(y);
			const float* cr = c->image.ptr<float>(y);
			const cv::Vec3b* gr = haveColor ? g->image.ptr<cv::Vec3b>(y) : 0;
			for(int x = 0; x < d->image.cols; x += pixelStep_)
			{
				const float z = dr[x];
				if(!std::isfinite(z) || z < minDepth_ || z > maxDepth_) continue;
				if(cr[x] < confThreshold_) continue;

				Eigen::Vector3d p((x - cx) * z / fx, (y - cy) * z / fy, z);
				Eigen::Vector3d w = R * p + t;
				if(!w.allFinite()) continue;

				PointT pt;
				pt.x = (float)w.x(); pt.y = (float)w.y(); pt.z = (float)w.z();
				if(gr) { const cv::Vec3b& c = gr[x];      // OpenCV order is BGR
				         pt.b = c[0]; pt.g = c[1]; pt.r = c[2]; }
				else   { pt.r = pt.g = pt.b = 200; }
				kf->push_back(pt);
			}
		}
		if(kf->empty()) return;

		if(pubKfCloud_.getNumSubscribers() > 0)
		{
			sensor_msgs::PointCloud2 out;
			pcl::toROSMsg(*kf, out);
			out.header.frame_id = "world";
			out.header.stamp = msg->depth.header.stamp;
			pubKfCloud_.publish(out);
		}

		{
			std::unique_lock<std::mutex> lk(mapMtx_);
			*global_ += *kf;
			// Voxel-filter as we go. Filtering every frame is wasteful, so only
			// once enough new points have accumulated to matter.
			pending_ += kf->size();
			if(pending_ > 200000)
			{
				CloudT::Ptr f(new CloudT);
				voxel_.setInputCloud(global_);
				voxel_.filter(*f);
				global_.swap(f);
				pending_ = 0;
				// hard cap: a voxel grid still grows as the corridor gets longer
				if((int)global_->size() > maxPoints_)
				{
					resolution_ *= 1.3;
					voxel_.setLeafSize(resolution_, resolution_, resolution_);
					ROS_WARN("dense_mapping: %zu pts over cap, coarsening voxel to %.3f m",
					         global_->size(), resolution_);
				}
			}
			++nFused_;
		}
	}

	void onTimer(const ros::TimerEvent&)
	{
		CloudT::Ptr snapshot;
		{
			std::unique_lock<std::mutex> lk(mapMtx_);
			if(global_->empty()) return;
			snapshot = CloudT::Ptr(new CloudT(*global_));
		}
		sensor_msgs::PointCloud2 out;
		pcl::toROSMsg(*snapshot, out);
		out.header.frame_id = "world";
		out.header.stamp = ros::Time::now();
		pubCloud_.publish(out);
		ROS_INFO_THROTTLE(10.0, "dense_mapping: %zu pts from %d keyframes "
		                  "(%d dropped, voxel %.3f m)",
		                  snapshot->size(), nFused_, nDropped_, resolution_);
	}

	double resolution_, confThreshold_, minDepth_, maxDepth_, publishPeriod_;
	int pixelStep_, maxPoints_;
	int nFused_ = 0, nDropped_ = 0;
	size_t pending_ = 0;

	CloudT::Ptr global_;
	pcl::VoxelGrid<PointT> voxel_;

	std::deque<polytunnel_vio::DepthMsg::ConstPtr> queue_;
	std::mutex mtx_, mapMtx_;
	std::condition_variable cv_;
	std::thread worker_;
	bool stop_ = false;

	ros::Subscriber sub_;
	ros::Publisher pubCloud_, pubKfCloud_;
	ros::Timer timer_;
};

int main(int argc, char** argv)
{
	ros::init(argc, argv, "dense_mapping");
	ros::NodeHandle nh, pnh("~");
	DenseMapper mapper(nh, pnh);
	ros::spin();
	return 0;
}
