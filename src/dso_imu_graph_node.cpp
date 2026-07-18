/**
* dso_imu_graph: loosely-coupled fusion of mono DSO odometry with IMU preintegration
* in a GTSAM iSAM2 factor graph.
*
* Design (OKVIS2-inspired, but loosely coupled):
*  - DSO (run in pure mono mode) emits camera poses in an arbitrary-scale visual world.
*    Consecutive keyframe-relative poses become "scaled between" factors: rotation used
*    as-is, translation multiplied by exp(s) where s is a per-DSO-session log-scale state.
*  - IMU measurements between keyframes become gtsam::CombinedImuFactor (Forster
*    preintegration with full covariance propagation - the math VI-DSO lacked).
*  - The graph persists across DSO re-initializations: biases, velocity and gravity
*    survive; each DSO session gets a fresh scale key seeded from the previous estimate.
*
* Inputs:  /vi_dso/pose (geometry_msgs/PoseStamped, from vi_dso_ros in mode:=mono)
*          /ms/imu/data (sensor_msgs/Imu)
* Outputs: /dso_graph/odom, /dso_graph/path  (metric, gravity-aligned world)
*          /dso_graph/scale (std_msgs/Float64, current metric scale of DSO's world)
*/

#include <deque>
#include <fstream>
#include <sstream>
#include <cmath>

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <std_msgs/Float64.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/base/numericalDerivative.h>

using namespace gtsam;
using symbol_shorthand::X;   // Pose3
using symbol_shorthand::V;   // Vector3 velocity
using symbol_shorthand::B;   // imuBias::ConstantBias
using symbol_shorthand::S;   // double log-scale (one per DSO session)


// Relative-pose factor from up-to-scale odometry: rotation as measured,
// translation scaled by exp(s). Numerical Jacobians (keyframe rate is low).
class ScaledBetweenFactor : public NoiseModelFactorN<Pose3, Pose3, double>
{
	Rot3 Rm_;
	Vector3 tm_;   // measured relative translation, visual (unscaled) units
public:
	ScaledBetweenFactor(Key i, Key j, Key s, const Rot3& Rm, const Vector3& tm,
	                    const SharedNoiseModel& model)
		: NoiseModelFactorN<Pose3, Pose3, double>(model, i, j, s), Rm_(Rm), tm_(tm) {}

	Vector evaluateError(const Pose3& Xi, const Pose3& Xj, const double& s,
	                     OptionalMatrixType H1, OptionalMatrixType H2,
	                     OptionalMatrixType H3) const override
	{
		auto err = [this](const Pose3& a, const Pose3& b, double sc) -> Vector6
		{
			Pose3 rel = a.between(b);
			Vector6 e;
			e.head<3>() = Rot3::Logmap(Rm_.between(rel.rotation()));
			e.tail<3>() = rel.translation() - std::exp(sc) * tm_;
			return e;
		};
		if(H1) *H1 = numericalDerivative31<Vector6, Pose3, Pose3, double>(err, Xi, Xj, s);
		if(H2) *H2 = numericalDerivative32<Vector6, Pose3, Pose3, double>(err, Xi, Xj, s);
		if(H3) *H3 = numericalDerivative33<Vector6, Pose3, Pose3, double>(err, Xi, Xj, s);
		return err(Xi, Xj, s);
	}
};


// ------------------------------------------------------------------ state
ISAM2* isam = 0;
NonlinearFactorGraph newFactors;
Values newValues;

std::shared_ptr<PreintegratedCombinedMeasurements::Params> pimParams;
PreintegratedCombinedMeasurements* pim = 0;

std::deque<sensor_msgs::Imu> imuBuf;

Pose3 T_BC;                  // imu <- cam extrinsic
bool haveInit = false;
int kfIdx = -1;              // current keyframe index
int sessIdx = 0;             // DSO session (increments on DSO re-init)
double lastKfTime = -1;
Pose3 lastKfDsoPose;         // DSO camera pose (visual world) at last keyframe
Pose3 lastDsoPose;           // most recent DSO pose (reset detection)
bool haveLastDso = false;
double lastDsoTime = -1;

NavState lastState;
imuBias::ConstantBias lastBias;
double lastLogScale = 0;

// params
double kfDt = 0.5;           // min seconds between keyframes
double odomSigmaRot = 0.01;  // rad, per keyframe interval
double odomSigmaTrans = 0.03;// visual-units-ish; scaled residual, keep loose
double scalePriorSigma = 2.3;// log-scale prior sigma (~10x uncertainty)
double scaleRwSigma = 0.002; // log-scale random walk per sqrt(s). A/B tested: 0.01 destabilizes
                             // (scale follows IMU noise in unobservable stretches), 0.002 ~= constant
double odomHuber = 1.345;    // Huber k on the odometry factor; 0 disables (suspect: clips turn intervals)
bool metricInput = false;    // true = input DSO pose is already METRIC (stereo): no scale state,
                             // plain BetweenFactor<Pose3> odometry (LIO-SAM principle — a metric
                             // measurement enters the graph, so nothing to diverge)
double resetJumpThresh = 1.0;// visual units between consecutive DSO poses => reset

ros::Publisher pubOdom, pubPath, pubScale, pubCloud, pubDsoAligned;
nav_msgs::Path path;
std::vector<std::pair<double, Pose3>> dsoKfPoses;   // DSO pose at each keyframe (visual world)
std::vector<double> kfTimes;                        // keyframe timestamps by index

// latest alignment between DSO's visual world and the metric graph world,
// refreshed every keyframe: p_metric = X_kf * ( exp(s) * (T_dsoworld_body_kf^-1 * p_dso) )
Pose3 alignDsoPose;    // imu pose in DSO world at last keyframe
Pose3 alignGraphPose;  // same instant, metric world
bool haveAlign = false;


Pose3 poseFromMsg(const geometry_msgs::PoseStamped& m)
{
	return Pose3(Rot3(Quaternion(m.pose.orientation.w, m.pose.orientation.x,
	                             m.pose.orientation.y, m.pose.orientation.z)),
	             Point3(m.pose.position.x, m.pose.position.y, m.pose.position.z));
}


// gravity-aligned initial orientation from mean accelerometer (roll/pitch only)
Rot3 gravityAlignedR(const Vector3& accMean)
{
	Vector3 g_b = accMean.normalized();          // measured specific force ~ -gravity in body
	Vector3 z_w(0, 0, 1);
	Vector3 v = g_b.cross(z_w);
	double sinT = v.norm(), cosT = g_b.dot(z_w);
	if(sinT < 1e-8) return Rot3();
	Vector3 axis = v / sinT;
	return Rot3::AxisAngle(Unit3(axis), std::atan2(sinT, cosT)).inverse();
}


void publishFusedMap(const Values& est, const ros::Time& stamp);

geometry_msgs::PoseStamped toMsg(const Pose3& P, const ros::Time& stamp)
{
	geometry_msgs::PoseStamped ps;
	ps.header.stamp = stamp;
	ps.header.frame_id = "world";
	Quaternion q = P.rotation().toQuaternion();
	ps.pose.position.x = P.x();
	ps.pose.position.y = P.y();
	ps.pose.position.z = P.z();
	ps.pose.orientation.w = q.w();
	ps.pose.orientation.x = q.x();
	ps.pose.orientation.y = q.y();
	ps.pose.orientation.z = q.z();
	return ps;
}


// publish current state + FULL smoothed path rebuilt from the latest estimate
// (iSAM2 revises past states every update; appending incremental poses shows
// stale mixed-epoch estimates and looks jagged)
void publishState(const ros::Time& stamp, const Values& est)
{
	nav_msgs::Odometry od;
	od.header.stamp = stamp;
	od.header.frame_id = "world";
	od.child_frame_id = "imu";
	od.pose.pose = toMsg(lastState.pose(), stamp).pose;
	od.twist.twist.linear.x = lastState.velocity().x();
	od.twist.twist.linear.y = lastState.velocity().y();
	od.twist.twist.linear.z = lastState.velocity().z();
	pubOdom.publish(od);

	path.header.stamp = stamp;
	path.header.frame_id = "world";
	path.poses.clear();
	for(int k = 0; k <= kfIdx; k++)
		if(est.exists(X(k)))
		{
			ros::Time ks = (k < (int)kfTimes.size()) ? ros::Time(kfTimes[k]) : stamp;
			path.poses.push_back(toMsg(est.at<Pose3>(X(k)), ks));
		}
	pubPath.publish(path);

	// DSO's own trajectory mapped into the metric world with the current
	// alignment + scale: directly comparable to the fused (orange) path
	if(haveAlign && !dsoKfPoses.empty())
	{
		nav_msgs::Path dp;
		dp.header = path.header;
		double s = std::exp(lastLogScale);
		Pose3 dsoInv = alignDsoPose.inverse();
		for(const auto& kv : dsoKfPoses)
		{
			Pose3 rel = dsoInv * kv.second;                       // relative to alignment anchor
			Pose3 metric(alignGraphPose.rotation() * rel.rotation(),
			             alignGraphPose.transformFrom(Point3(s * rel.translation())));
			dp.poses.push_back(toMsg(metric, stamp));
		}
		pubDsoAligned.publish(dp);
	}

	std_msgs::Float64 sc;
	sc.data = std::exp(lastLogScale);
	pubScale.publish(sc);

	publishFusedMap(est, stamp);
}


void integrateImuUpTo(double tEnd)
{
	while(!imuBuf.empty())
	{
		const sensor_msgs::Imu& m = imuBuf.front();
		double t = m.header.stamp.toSec();
		if(t > tEnd) break;
		static double tPrev = -1;
		double dt = (tPrev < 0) ? 1.0/200.0 : (t - tPrev);
		tPrev = t;
		if(dt > 0 && dt < 0.1)
			pim->integrateMeasurement(
				Vector3(m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z),
				Vector3(m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z), dt);
		imuBuf.pop_front();
	}
}


void imuCb(const sensor_msgs::ImuConstPtr& msg)
{
	imuBuf.push_back(*msg);
	if(imuBuf.size() > 40000) imuBuf.pop_front();
}


// per-keyframe camera-frame clouds from DSO, keyed by stamp; the metric map is
// assembled in publishState using the graph's own corrected poses
std::map<double, std::vector<float>> kfClouds;

void kfCloudCb(const sensor_msgs::PointCloud2ConstPtr& msg)
{
	if(msg->point_step != 12 || msg->fields.size() != 3) return;
	std::vector<float>& v = kfClouds[msg->header.stamp.toSec()];
	v.resize(msg->width * 3);
	memcpy(v.data(), msg->data.data(), msg->width * 12);
	if(kfClouds.size() > 3000) kfClouds.erase(kfClouds.begin());
}


// assemble the metric map: every DSO keyframe's points transformed by the
// closest GRAPH keyframe's fused pose (not DSO's own pose)
void publishFusedMap(const Values& est, const ros::Time& stamp)
{
	if(pubCloud.getNumSubscribers() == 0 || kfTimes.empty()) return;

	double s = std::exp(lastLogScale);
	size_t total = 0;
	for(auto& kv : kfClouds) total += kv.second.size() / 3;

	sensor_msgs::PointCloud2 pc;
	pc.header.stamp = stamp;
	pc.header.frame_id = "world";
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
	pc.data.reserve(total * 12);

	size_t n = 0;
	for(auto& kv : kfClouds)
	{
		// nearest graph keyframe in time
		auto it = std::lower_bound(kfTimes.begin(), kfTimes.end(), kv.first);
		int k = (int)(it - kfTimes.begin());
		if(k > 0 && (k >= (int)kfTimes.size() || std::abs(kfTimes[k-1]-kv.first) < std::abs(kfTimes[k]-kv.first))) k--;
		if(k >= (int)kfTimes.size() || std::abs(kfTimes[k]-kv.first) > 0.6) continue;
		if(!est.exists(X(k))) continue;
		Pose3 Xk = est.at<Pose3>(X(k));
		double sk = est.exists(S(k)) ? std::exp(est.at<double>(S(k))) : s;   // per-KF scale

		const std::vector<float>& v = kv.second;
		for(size_t i = 0; i + 2 < v.size(); i += 3)
		{
			// camera point (visual units) -> metric camera -> imu -> world
			Point3 pc_cam(sk*v[i], sk*v[i+1], sk*v[i+2]);
			Point3 q = Xk.transformFrom(T_BC.transformFrom(pc_cam));
			float xyz[3] = {(float)q.x(), (float)q.y(), (float)q.z()};
			pc.data.insert(pc.data.end(), (uint8_t*)xyz, (uint8_t*)xyz + 12);
			n++;
		}
	}
	pc.width = n;
	pc.row_step = 12 * n;
	pubCloud.publish(pc);
}


void dsoCb(const geometry_msgs::PoseStampedConstPtr& msg)
{
	double t = msg->header.stamp.toSec();
	Pose3 T_wd_c = poseFromMsg(*msg);            // camera in DSO visual world
	Pose3 T_wd_b = T_wd_c * T_BC.inverse();      // imu in DSO visual world

	// ---- DSO reset detection: big jump between consecutive DSO poses
	bool dsoReset = false;
	if(haveLastDso)
	{
		double jump = (T_wd_b.translation() - lastDsoPose.translation()).norm();
		double dtm = t - lastDsoTime;
		if(jump > resetJumpThresh || dtm < 0)
			dsoReset = true;
	}
	lastDsoPose = T_wd_b;
	lastDsoTime = t;
	haveLastDso = true;

	// ---- initialization: first DSO pose + enough IMU for gravity
	if(!haveInit)
	{
		if(imuBuf.size() < 100) return;   // ~0.5 s of IMU
		Vector3 accMean = Vector3::Zero();
		int n = 0;
		for(const auto& m : imuBuf)
		{
			if(m.header.stamp.toSec() > t) break;
			accMean += Vector3(m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z);
			n++;
		}
		if(n < 50) return;
		accMean /= n;

		kfIdx = 0;
		kfTimes.push_back(t);
		Pose3 X0(gravityAlignedR(accMean), Point3(0,0,0));
		lastState = NavState(X0, Vector3::Zero());
		lastBias = imuBias::ConstantBias();
		lastLogScale = 0;

		newValues.insert(X(0), X0);
		newValues.insert(V(0), Vector3(Vector3::Zero()));
		newValues.insert(B(0), lastBias);
		if(!metricInput)
			newValues.insert(S(0), lastLogScale);   // scale is per-KEYFRAME (random-walk chain)

		// gauge prior: position + yaw anchored; roll/pitch stiffness is a parameter —
		// too tight locks in a motion-polluted gravity guess, too loose lets
		// attitude/scale/bias wander together (our vision is only up-to-scale)
		double attSigma;
		ros::NodeHandle("~").param<double>("att_prior_sigma", attSigma, 0.1);
		newFactors.addPrior(X(0), X0, noiseModel::Diagonal::Sigmas(
			(Vector6() << attSigma, attSigma, 0.01, 0.001, 0.001, 0.001).finished()));
		newFactors.addPrior(V(0), Vector3(Vector3::Zero()), noiseModel::Isotropic::Sigma(3, 1.0));
		newFactors.addPrior(B(0), lastBias, noiseModel::Diagonal::Sigmas(
			(Vector6() << 0.1,0.1,0.1, 0.01,0.01,0.01).finished()));            // acc then gyro
		if(!metricInput)
			newFactors.addPrior(S(0), 0.0, noiseModel::Isotropic::Sigma(1, scalePriorSigma));

		isam->update(newFactors, newValues);
		newFactors.resize(0);
		newValues.clear();

		// drop IMU older than the first keyframe
		while(!imuBuf.empty() && imuBuf.front().header.stamp.toSec() <= t) imuBuf.pop_front();
		pim = new PreintegratedCombinedMeasurements(pimParams, lastBias);

		lastKfTime = t;
		lastKfDsoPose = T_wd_b;
		haveInit = true;
		ROS_INFO("dso_imu_graph: initialized (gravity-aligned, scale prior e^0)");
		return;
	}

	// ---- keyframe policy
	double rotDeg = Rot3::Logmap(lastKfDsoPose.rotation().between(T_wd_b.rotation())).norm() * 57.3;
	if(t - lastKfTime < kfDt && rotDeg < 10.0 && !dsoReset) return;
	if(t <= lastKfTime) return;

	if(dsoReset)
	{
		sessIdx++;
		ROS_WARN("dso_imu_graph: DSO re-initialization detected -> session %d "
		         "(scale chain re-seeded at %.3f; biases/velocity persist)", sessIdx, std::exp(lastLogScale));
		dsoKfPoses.clear();   // old session lives in a different visual world
	}

	// ---- IMU factor between keyframes
	integrateImuUpTo(t);
	if(pim->deltaTij() < 1e-4)
	{
		// no IMU between frames (shouldn't happen at 200 Hz) - skip this KF
		lastKfTime = t;
		lastKfDsoPose = T_wd_b;
		return;
	}

	int i = kfIdx, j = kfIdx + 1;
	NavState pred = pim->predict(lastState, lastBias);
	newValues.insert(X(j), pred.pose());
	newValues.insert(V(j), Vector3(pred.velocity()));
	newValues.insert(B(j), lastBias);
	if(!metricInput)
		newValues.insert(S(j), lastLogScale);
	newFactors.add(CombinedImuFactor(X(i), V(i), X(j), V(j), B(i), B(j), *pim));

	// ---- odometry factor (+ scale chain, mono only)
	if(!dsoReset)
	{
		Pose3 rel = lastKfDsoPose.between(T_wd_b);
		SharedNoiseModel noise = noiseModel::Diagonal::Sigmas(
			(Vector6() << odomSigmaRot, odomSigmaRot, odomSigmaRot,
			              odomSigmaTrans, odomSigmaTrans, odomSigmaTrans).finished());
		if(odomHuber > 0)
			noise = noiseModel::Robust::Create(
				noiseModel::mEstimator::Huber::Create(odomHuber),
				std::static_pointer_cast<noiseModel::Base>(noise));

		if(metricInput)
		{
			// stereo: metric relative pose, no scale state. The robust loss lets
			// the graph reject stereo motion in far-structure sections where the
			// short baseline collapses, coasting on the IMU instead.
			newFactors.add(BetweenFactor<Pose3>(X(i), X(j), rel, noise));
		}
		else
		{
			// mono: scale-free vision → estimate a per-KF scale (random-walk chain)
			double dtk = t - kfTimes.back();
			newFactors.add(BetweenFactor<double>(S(i), S(j), 0.0,
				noiseModel::Isotropic::Sigma(1, scaleRwSigma * std::sqrt(std::max(dtk, 0.05)))));
			newFactors.add(ScaledBetweenFactor(X(i), X(j), S(i), rel.rotation(), rel.translation(), noise));
		}
	}
	else if(!metricInput)
	{
		// new DSO session: fresh visual world, restart the scale chain
		newFactors.addPrior(S(j), lastLogScale, noiseModel::Isotropic::Sigma(1, 0.7));
	}

	// ---- optimize
	isam->update(newFactors, newValues);
	newFactors.resize(0);
	newValues.clear();
	Values est = isam->calculateEstimate();

	kfIdx = j;
	kfTimes.push_back(t);
	lastState = NavState(est.at<Pose3>(X(j)), est.at<Vector3>(V(j)));
	lastBias = est.at<imuBias::ConstantBias>(B(j));
	if(!metricInput)
		lastLogScale = est.at<double>(S(j));   // stays 0 (scale 1) for metric stereo input

	pim->resetIntegrationAndSetBias(lastBias);
	lastKfTime = t;
	lastKfDsoPose = T_wd_b;

	alignDsoPose = T_wd_b;
	alignGraphPose = lastState.pose();
	haveAlign = true;
	dsoKfPoses.push_back(std::make_pair(t, T_wd_b));

	publishState(msg->header.stamp, est);
	ROS_INFO_THROTTLE(2.0, "dso_imu_graph: kf %d sess %d | scale %.3f | v %.2f m/s | bias_a norm %.3f",
	                  kfIdx, sessIdx, std::exp(lastLogScale),
	                  lastState.velocity().norm(), lastBias.accelerometer().norm());
}


// same IMU_info.txt as vi_dso_ros: 3 rows of T_BC (imu <- cam), skipped 4th row,
// then gyr_n, acc_n, gyr_w, acc_w
void loadImuInfo(const std::string& fn, double& gn, double& an, double& gw, double& aw)
{
	std::ifstream inf(fn.c_str());
	if(!inf.good()) { ROS_FATAL("cannot open imu_info %s", fn.c_str()); exit(1); }
	Matrix3 R;
	Vector3 tr;
	std::string sline;
	int line = 0;
	double noise[4];
	while(line < 3 && std::getline(inf, sline))
	{
		std::istringstream ss(sline);
		for(int c = 0; c < 3; ++c) ss >> R(line, c);
		ss >> tr(line);
		++line;
	}
	std::getline(inf, sline); ++line;
	while(line < 8 && std::getline(inf, sline))
	{
		std::istringstream ss(sline);
		ss >> noise[line-4];
		++line;
	}
	T_BC = Pose3(Rot3(R), Point3(tr));
	gn = noise[0]; an = noise[1]; gw = noise[2]; aw = noise[3];
}


int main(int argc, char** argv)
{
	ros::init(argc, argv, "dso_imu_graph");
	ros::NodeHandle nh;
	ros::NodeHandle pnh("~");

	std::string imuInfoFile, dsoTopic, imuTopic;
	double gNorm;
	pnh.param<std::string>("imu_info", imuInfoFile, "");
	pnh.param<std::string>("dso_pose_topic", dsoTopic, "/vi_dso/pose");
	pnh.param<std::string>("imu_topic", imuTopic, "/ms/imu/data");
	pnh.param<double>("g_norm", gNorm, 9.81);
	pnh.param<double>("kf_dt", kfDt, 0.5);
	pnh.param<double>("odom_sigma_rot", odomSigmaRot, 0.01);
	pnh.param<double>("odom_sigma_trans", odomSigmaTrans, 0.03);
	pnh.param<double>("scale_prior_sigma", scalePriorSigma, 2.3);
	pnh.param<bool>("metric_input", metricInput, false);
	pnh.param<double>("scale_rw_sigma", scaleRwSigma, 0.002);
	pnh.param<double>("odom_huber", odomHuber, 1.345);
	pnh.param<double>("reset_jump_thresh", resetJumpThresh, 1.0);
	// OKVIS-style noise inflation: bench-calibrated densities are far too
	// optimistic on a vibrating platform (OKVIS2 configs run 5-20x inflated)
	double accInflate, gyrInflate;
	pnh.param<double>("acc_sigma_inflation", accInflate, 5.0);   // sweep winner (config B)
	pnh.param<double>("gyro_sigma_inflation", gyrInflate, 2.0);
	if(imuInfoFile.empty()) { ROS_FATAL("~imu_info required"); return 1; }

	double gn, an, gw, aw;
	loadImuInfo(imuInfoFile, gn, an, gw, aw);
	an *= accInflate;
	gn *= gyrInflate;
	ROS_INFO("inflated sigmas: gyr %.4f acc %.4f (x%.0f / x%.0f)", gn, an, gyrInflate, accInflate);
	ROS_INFO_STREAM("T_BC:\n" << T_BC.matrix());
	ROS_INFO("IMU noise: gyr %.2e acc %.2e gyr_w %.2e acc_w %.2e", gn, an, gw, aw);

	pimParams = PreintegratedCombinedMeasurements::Params::MakeSharedU(gNorm);
	pimParams->setAccelerometerCovariance(Matrix3::Identity() * an * an);
	pimParams->setGyroscopeCovariance(Matrix3::Identity() * gn * gn);
	pimParams->setIntegrationCovariance(Matrix3::Identity() * 1e-8);
	pimParams->setBiasAccCovariance(Matrix3::Identity() * aw * aw);
	pimParams->setBiasOmegaCovariance(Matrix3::Identity() * gw * gw);
	pimParams->setBiasAccOmegaInit(Matrix6::Identity() * 1e-5);

	ISAM2Params ip;
	ip.relinearizeThreshold = 0.1;
	ip.relinearizeSkip = 1;
	isam = new ISAM2(ip);

	path.header.frame_id = "world";
	pubOdom  = nh.advertise<nav_msgs::Odometry>("dso_graph/odom", 50);
	pubPath  = nh.advertise<nav_msgs::Path>("dso_graph/path", 5);
	pubScale = nh.advertise<std_msgs::Float64>("dso_graph/scale", 5);
	pubCloud = nh.advertise<sensor_msgs::PointCloud2>("dso_graph/cloud", 2);
	pubDsoAligned = nh.advertise<nav_msgs::Path>("dso_graph/dso_path_aligned", 5);

	ros::Subscriber s1 = nh.subscribe(imuTopic, 2000, &imuCb);
	ros::Subscriber s2 = nh.subscribe(dsoTopic, 50, &dsoCb);
	ros::Subscriber s3 = nh.subscribe("/vi_dso/kf_cloud", 30, &kfCloudCb);

	ROS_INFO("dso_imu_graph ready: fusing %s + %s", dsoTopic.c_str(), imuTopic.c_str());
	ros::spin();
	return 0;
}
