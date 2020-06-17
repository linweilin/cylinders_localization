// ROS
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf/transform_broadcaster.h>


#include <boost/thread/thread.hpp>
#include <vector>
#include <iostream>
#include <string>
#include <fstream>

#include "parameter_reader.h" // for parameterReader.txt
#include "line.hpp" // Define a class Line derived from points using Eigen

// PCL
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/common_headers.h>
#include <pcl/common/intersections.h>
#include <pcl/console/parse.h>
#include <pcl/common/distances.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
// Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/ceres.h>

using namespace std;

sensor_msgs::PointCloud2 transformed_pc_msg;
tf::Transform tf_transform;

// pose to optimize
double parameters[7] = {0, 0, 0, 1, 0, 0, 0};
Eigen::Map<Eigen::Quaterniond> q_w_curr(parameters);
Eigen::Map<Eigen::Vector3d> t_w_curr(parameters + 4);

const int poles_num_in_map = 16; // refer to the number of pole in file bim_vector_map.txt
Eigen::Matrix4d initial_transform_matrix; // initial pose from UWB or coders of robot(what ever)

// print euler angles (unit: degree)
void PrintEulerAnglesDegreeAndTranslation (Eigen::Matrix4d matrix)
{
    Eigen::Vector3d trans = matrix.block(0,3,3,1);
    Eigen::Matrix3d block = matrix.block(0,0,3,3);
    Eigen::Vector3d euler_angles = block.eulerAngles(0, 1, 2);
    Eigen::Vector3d euler_angles_degree = euler_angles * 180 / M_PI; // rad to degree;
    std::cout << "Translation = " << trans.transpose() << "\n"
        << "Euler angles (degree) = " << euler_angles_degree.transpose() << "\n" << std::endl;
}

// read initial pose from file
Eigen::Matrix4d ReadData()
{
    ifstream fin("./initial_pose.txt");
    if (!fin)
    {
        cerr<<"please run in the directory included initial_pose.txt!"<<endl;
        exit(0);
    }
    std::cout << "Initial pose : " << "\n" << initial_transform_matrix << '\n' << std::endl;
    PrintEulerAnglesDegreeAndTranslation(initial_transform_matrix);
    
    Eigen::Matrix4d matrix;
    Eigen::Vector4d v;
    for (int j = 0; j < 4; j++)
    {
        for (int i = 0; i < 4; i++)
        {
            fin >> v(i);
        }
        matrix.row(j) = v;
    }
    fin.close();
    
    return matrix;
}


// 相机坐标系下杆件轴线坐标点变换到初始位姿
void LineAssociateToMap(std::vector<Line> lines_in_cam, std::vector<Line>& poles_from_cam_to_map)
{
    initial_transform_matrix = ReadData();
    Eigen::Matrix3d initial_rotation_matrix = initial_transform_matrix.block(0,0,3,3);
    Eigen::Vector3d initial_translation = initial_transform_matrix.block(0,3,3,1);
//     cout << "initial_rotation_matrix is: \n" << initial_rotation_matrix << endl;
//     cout << "initial_translation is: " << initial_translation.transpose() << endl;
//     Eigen::Quaterniond quat (initial_rotation_matrix);
    
    // Transformed to local coordinate
//     poles_from_cam_to_map.reserve(lines_in_cam.size());
    for (int i = 0; i < lines_in_cam.size(); i++)
    {

        cout << i << " before transformed: "; lines_in_cam.at(i).PrintLine();

        // 因为是对点的坐标变换，先旋转再平移和先平移再旋转无区别
        // P^I = P^I_C * P^C
        Line l = lines_in_cam.at(i);
        Eigen::Vector3d transformed_pts = initial_translation + initial_rotation_matrix * l.GetStartPoint();
        Eigen::Vector3d transformed_pte = initial_translation + initial_rotation_matrix * l.GetEndPoint();
        l.SetPoints(transformed_pts, transformed_pte);
//         lines_in_cam.at(i) = l;
        poles_from_cam_to_map.push_back(l);
//         cout << i << " after transformed: "; lines_in_cam.at(i).PrintLine();
        cout << i << " after transformed: "; poles_from_cam_to_map.at(i).PrintLine();
    }
//     cout << '\n';
}

void CylinderRecognition(std::vector<Line> &lines_in_cam, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    // declare ParameterReader class object
    ParameterReader pd;

//     std::cerr << "PointCloud before filtering: " << cloud->points.size () << " data points." << "\n\n";
// 
//     pcl::visualization::CloudViewer viewer1("01 Input pointcloud");
//     viewer1.showCloud(cloud,"1");
//     while (!viewer1.wasStopped ())
//     {}

    //*********** Voxel grid downsample the dataset using a leaf size of 1cm ************//
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    float leafsize = atof( pd.getData( "leaf_size" ).c_str());
//     std::cout<<"leaf_size : "<< leafsize << std::endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_voxelgrid_filtered (new pcl::PointCloud<pcl::PointXYZ>);
    vg.setInputCloud (cloud);
    vg.setLeafSize (leafsize, leafsize, leafsize);
    vg.filter (*cloud_voxelgrid_filtered);
//     std::cout << "PointCloud after VoxselGrid filtering has: " << cloud_voxelgrid_filtered->points.size ()  << " points." << std::endl;
// 
//     pcl::visualization::CloudViewer viewer2("02 Cloud after VoxselGrid filtering");
//     viewer2.showCloud(cloud,"1");
//     while (!viewer2.wasStopped ())
//     {}

    /**
     * statistical_removal for removing outliers
     */
    // Create the filtering object
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud (cloud_voxelgrid_filtered);
    float meanK( atof( pd.getData( "mean_k" ).c_str() ) );
    sor.setMeanK (meanK);
    float stdDevMulThresh( atof( pd.getData( "std_dev_mul_thresh").c_str() ) );
    sor.setStddevMulThresh (stdDevMulThresh);
    sor.setNegative (true);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_remain (new pcl::PointCloud<pcl::PointXYZ>);
    sor.filter (*cloud_remain);

//     std::cerr << "Cloud_outliers after filtering: " << std::endl;
//     std::cerr << *cloud_remain << std::endl;
//   
//     pcl::visualization::CloudViewer viewer3("03 Cloud_outliers after filtering");
//     viewer3.showCloud(cloud_remain,"1");
//     while (!viewer3.wasStopped ())
//     {}
  
    sor.setNegative (false);
    sor.filter (*cloud_remain);
  
//     std::cerr << "Cloud_inliers after filtering: " << std::endl;
//     std::cerr << *cloud_remain << std::endl;
//   
//     pcl::visualization::CloudViewer viewer4("04 Cloud_inliers after filtering");
//     viewer4.showCloud(cloud_remain,"1");
//     while (!viewer4.wasStopped ())
//     {}
  
    // Estimate point normals
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;    // normal estimate object
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
    ne.setSearchMethod (tree);
    ne.setInputCloud (cloud_remain);
    int kSearch ( atoi( pd.getData( "k_search" ).c_str() ) );
    ne.setKSearch (kSearch);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
    ne.compute (*cloud_normals);
  

    /**
     * Normals visualization
     */
//     boost::shared_ptr<pcl::visualization::PCLVisualizer> Normals_viewer (new pcl::visualization::PCLVisualizer ("05 Normal Viewer"));
//     Normals_viewer->setBackgroundColor (0, 0, 0);
//     Normals_viewer->addCoordinateSystem (0.1); // coordinate system unit length
//     Normals_viewer->addPointCloud<pcl::PointXYZ> (cloud_remain, "sample cloud");
//     Normals_viewer->addPointCloudNormals<pcl::PointXYZ,pcl::Normal> (cloud_remain, cloud_normals, 50, 0.1, "normal");
//     Normals_viewer->initCameraParameters ();
//     while(!Normals_viewer->wasStopped())
//     {
//         Normals_viewer->spinOnce(100);
//         boost::this_thread::sleep (boost::posix_time::microseconds (100000));
//     }
    
    // for plane segmentation
/*
    pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> seg;     // segment object
    // Create the segmentation object for the planar model and set all the parameters
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_NORMAL_PLANE);
    float pextNorDisWei ( atof ( pd.getData ("pext_nor_dis_wei").c_str() ) );
    seg.setNormalDistanceWeight (pextNorDisWei);
    seg.setMethodType (pcl::SAC_RANSAC);
    int pextMaxIter ( atoi ( pd.getData ("pext_max_iter").c_str() ) );
    seg.setMaxIterations (pextMaxIter);
    float pextDisThres ( atof ( pd.getData ("pext_dis_thres").c_str() ) );
    seg.setDistanceThreshold (pextDisThres);
    pcl::ExtractIndices<pcl::PointXYZ> extract;  // points extraction object
    pcl::ExtractIndices<pcl::Normal> extract_normals; 

    // no need to specitfy length at initialization
    // std::vector will dynamically allocate memory for its contents as requested
    std::vector< pcl::PointCloud<pcl::PointXYZ> > vec_planes_clouds;
    std::vector< pcl::ModelCoefficients > vec_planes_coefficients;
    std::vector< pcl::PointCloud<pcl::PointXYZ>::Ptr > vec_planes_ptr;
    vec_planes_clouds.reserve(10);   // Set the capacity to (at least) 10
    vec_planes_coefficients.reserve(10) ;    // Set the capacity to (at least) 10
    vec_planes_ptr.reserve(10);    // Set the capacity to (at least) 10
    

    
    int nNumPlanes ( atoi ( pd.getData("nNum_planes").c_str() ) );
    for (int nPlanes = 0; nPlanes < nNumPlanes; nPlanes++)
    {
        seg.setInputCloud (cloud_remain);
        seg.setInputNormals (cloud_normals);
        
        // Obtain the plane inliers and coefficients
        pcl::PointIndices::Ptr inliers_plane (new pcl::PointIndices);    
        pcl::ModelCoefficients::Ptr coefficients_planes (new pcl::ModelCoefficients);
        seg.segment (*inliers_plane, *coefficients_planes);
        // Extract the planar inliers from the input cloud
        extract.setInputCloud (cloud_remain);
        extract.setIndices (inliers_plane);
        extract.setNegative (false);
        
        // Get the points associated with the plane surface
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_planes (new pcl::PointCloud<pcl::PointXYZ> ());
        extract.filter (*cloud_planes);
        
        // If there is not any plane
        if (inliers_plane->indices.size () == 0)
        {
            std::cout << "Can't find the plane component." << std::endl;
            break;
        }      
        else
        {
            std::cerr << "PointCloud representing the plane component[" << nPlanes << "]: "<< cloud_planes ->points.size () << " data points." << std::endl;
            std::cerr << "Plane[" << nPlanes << "] coefficients:" << *coefficients_planes << std::endl;
            pcl::visualization::CloudViewer viewer_planes_segmentation("06 cloud_planes");
            viewer_planes_segmentation.showCloud(cloud_planes,"1");
            while (!viewer_planes_segmentation.wasStopped ())
            {}
        }
        
        // store cloud_planes, coeffcients
        vec_planes_clouds.push_back(*cloud_planes);
        vec_planes_coefficients.push_back(*coefficients_planes);
        vec_planes_ptr.push_back(cloud_planes);

        // Remove the planar inliers, extract the rest and subtitube the input cloud
        extract.setNegative (true);
        extract.filter (*cloud_remain);

        extract_normals.setNegative (true);
        extract_normals.setInputCloud (cloud_normals);
        extract_normals.setIndices (inliers_plane);
        extract_normals.filter (*cloud_normals);
      
        pcl::visualization::CloudViewer viewer7("07 remaining pointcloud after remove all planes extracted");
        viewer7.showCloud(cloud_remain,"1");
        while (!viewer7.wasStopped ())
        {}
    }
*/
    
    /**
     * Cylinder segmentation
     */
    // instantiation the segmentation object and set all the parameters for cylinder segmentation
    pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> seg;     // segment object
    pcl::ExtractIndices<pcl::PointXYZ> extract;  // points extraction object
    pcl::ExtractIndices<pcl::Normal> extract_normals; 
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_CYLINDER);
    seg.setMethodType (pcl::SAC_RANSAC);
    float csegNorDisWei ( atof ( pd.getData("cseg_nor_dis_wei").c_str() ) );
    seg.setNormalDistanceWeight (csegNorDisWei);
    int csegMaxIter ( atof ( pd.getData("cseg_max_iter").c_str() ) );
    seg.setMaxIterations (csegMaxIter);
    float csegDisThres ( atof ( pd.getData("cseg_dis_thres").c_str() ) );
    seg.setDistanceThreshold (csegDisThres);
    float csegMinRad ( atof ( pd.getData("cseg_min_rad").c_str() ) );
    float csegMaxRad ( atof ( pd.getData("cseg_max_rad").c_str() ) );
    seg.setRadiusLimits (csegMinRad, csegMaxRad);
  
    // no need to specitfy length at initialization
    // std::vector will dynamically allocate memory for its contents as requested
    std::vector< pcl::PointCloud<pcl::PointXYZ> > vec_cylinders_clouds;
    std::vector< pcl::ModelCoefficients > vec_cylinders_coefficients;
    std::vector< pcl::PointCloud<pcl::PointXYZ>::Ptr > vec_cylinders_ptr;
    vec_cylinders_clouds.reserve(10);   // Set the capacity to (at least) 10
    vec_cylinders_coefficients.reserve(10) ;    // Set the capacity to (at least) 10
    vec_cylinders_ptr.reserve(10);    // Set the capacity to (at least) 10
    
    int nNumPoles ( atoi ( pd.getData("nNum_poles").c_str() ) );
    for (int nCount = 0; nCount < nNumPoles; nCount++)
    {
        seg.setInputCloud (cloud_remain);
        seg.setInputNormals (cloud_normals);
      
        // Obtain the cylinder inliers and coefficients from the input cloud
        pcl::PointIndices::Ptr inliers_cylinder (new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr coefficients_cylinder (new pcl::ModelCoefficients);
        seg.segment (*inliers_cylinder, *coefficients_cylinder);
        extract.setInputCloud (cloud_remain);
        extract.setIndices (inliers_cylinder);
        extract.setNegative (false);

        // Get the points associated with the cylinder surface
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cylinders (new pcl::PointCloud<pcl::PointXYZ> ());
        extract.filter (*cloud_cylinders);

        // If there is not any cylinder
        if (inliers_cylinder->indices.size () == 0)
        {
            std::cout << "Can't find the cylindrical component." << std::endl;
            continue;
        }      
        else
        {
//             std::cerr << "PointCloud representing the cylindrical component[" << nCount << "]: "<< cloud_cylinders ->points.size () << " data points." << std::endl;
//             std::cerr << "Cylinder[" << nCount << "] coefficients:" << *coefficients_cylinder << std::endl;
//             pcl::visualization::CloudViewer viewer_cylinders_segmentation("08 cloud_cylinders");
//             viewer_cylinders_segmentation.showCloud(cloud_cylinders,"1");
//             while (!viewer_cylinders_segmentation.wasStopped ())
//             {}
        }
        
        // store cloud_cylinders, coeffcients
        vec_cylinders_clouds.push_back(*cloud_cylinders);
        vec_cylinders_coefficients.push_back(*coefficients_cylinder);
        vec_cylinders_ptr.push_back(cloud_cylinders);

        // Remove the cylinder inliers, extract the rest and subtitube the input cloud
        extract.setNegative (true);
        extract.filter(*cloud_remain);

        // Remove the cylinder normals, extract the rest and subtitube the input normals
        extract_normals.setNegative (true);
        extract_normals.setInputCloud (cloud_normals);
        extract_normals.setIndices (inliers_cylinder);
        extract_normals.filter (*cloud_normals);

//         pcl::visualization::CloudViewer viewer9("09 remaining pointcloud after remove all cylinders extracted");
//         viewer9.showCloud(cloud_remain,"1");
//         while (!viewer9.wasStopped ())
//         {}

    }
    
    if (vec_cylinders_ptr.size() != nNumPoles)
    {
        std::cout << "cannot find all the poles you want!" << '\n'
            << "Please check your parameters to fix it!" << std::endl;
        exit(0);
    }
    
    /**
     * Caculate the OBB bounding box for obtain the length of the cylinders.
     * No need to specitfy length at initialization.
     * std::vector will dynamically allocate memory for its contents as requested.
     * To save the menmory consumption, set the initial capacity to (at least) 10
     */    
    for (int nIndex = 0; nIndex < vec_cylinders_ptr.size(); nIndex++)
    {
        pcl::MomentOfInertiaEstimation <pcl::PointXYZ> feature_extractor;
        feature_extractor.setInputCloud ( vec_cylinders_ptr.at(nIndex) );
        feature_extractor.compute ();

        pcl::PointXYZ min_point_OBB;
        pcl::PointXYZ max_point_OBB;
        pcl::PointXYZ position_OBB;
        Eigen::Matrix3f rotational_matrix_OBB;
        Eigen::Vector3f major_vector, middle_vector, minor_vector;
        Eigen::Vector3f mass_center;

        feature_extractor.getOBB (min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);
        feature_extractor.getEigenVectors (major_vector, middle_vector, minor_vector);
        
        Eigen::Vector3f position (position_OBB.x, position_OBB.y, position_OBB.z);
        Eigen::Quaternionf quat (rotational_matrix_OBB);
        
        /**
         * Bounding box visualization
         */
//         boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_bounding_box (new pcl::visualization::PCLVisualizer ("10 viewer_bounding_box"));
//         viewer_bounding_box->setBackgroundColor (0, 0, 0);
//         viewer_bounding_box->addCoordinateSystem (0.1);
//         viewer_bounding_box->initCameraParameters ();
//         viewer_bounding_box->addPointCloud<pcl::PointXYZ> (cloud, "original cloud");  // use the original inputCloud
//         viewer_bounding_box->addPointCloud<pcl::PointXYZ> (vec_cylinders_ptr.at(nIndex), "sample cloud");    // use the cylinders pointcloud
//         viewer_bounding_box->addCube (position, quat, max_point_OBB.x - min_point_OBB.x, max_point_OBB.y - min_point_OBB.y, max_point_OBB.z - min_point_OBB.z, "OBB");
        
        /**
         * Axixs and radii of cylinders visualization
         * Calculate the euclidean distance between the two given points in OBB
         * They are uesd approxinmatelt as points at the begingning point and ending point of the cylinder
         */
        Eigen::Vector3f pt_on_start_plane (min_point_OBB.x, min_point_OBB.y, min_point_OBB.z);     // Point on the start plane of OBB before transform to local coordinate
        Eigen::Vector3f pt_on_end_plane (max_point_OBB.x, min_point_OBB.y, min_point_OBB.z);     // Point on the end plane of OBB before transform to local coordinate
        
        // Transformed to local coordinate
        pt_on_start_plane = rotational_matrix_OBB * pt_on_start_plane + position;
        pt_on_end_plane = rotational_matrix_OBB * pt_on_end_plane + position;

        // Find two ends of the axis by Eigen method
        Eigen::Vector3f pt_on_axis(vec_cylinders_coefficients.at(nIndex).values[0],
                                   vec_cylinders_coefficients.at(nIndex).values[1],
                                   vec_cylinders_coefficients.at(nIndex).values[2]);
        Eigen::Vector3f axis_orientation(vec_cylinders_coefficients.at(nIndex).values[3],
                                         vec_cylinders_coefficients.at(nIndex).values[4],
                                         vec_cylinders_coefficients.at(nIndex).values[5]);
        Eigen::Hyperplane<float, 3> start_point_plane (major_vector, pt_on_start_plane);
        Eigen::Hyperplane<float, 3> end_point_plane (major_vector, pt_on_end_plane);
        Eigen::ParametrizedLine<float, 3> axis_line (pt_on_axis,axis_orientation);
        Eigen::Vector3d start_intersection_point = axis_line.intersectionPoint(start_point_plane).cast<double>();
        Eigen::Vector3d end_intersection_point = axis_line.intersectionPoint(end_point_plane).cast<double>();
        
        // visualize axis line
        pcl::PointXYZ pStartIntersec (start_intersection_point(0), start_intersection_point(1), start_intersection_point(2));
        pcl::PointXYZ pEndIntersec (end_intersection_point(0), end_intersection_point(1), end_intersection_point(2));
//         viewer_bounding_box->addArrow(pStartIntersec, pEndIntersec, 1.0, 0.0, 0.0, true,"Red axis");
        
        // Store lines
        Line temp_line(start_intersection_point,end_intersection_point);
        lines_in_cam.push_back(temp_line);

//         while(!viewer_bounding_box->wasStopped())
//         {
//             viewer_bounding_box->spinOnce (100);
//             boost::this_thread::sleep (boost::posix_time::microseconds (100000));
//         }
    }
}

std::vector<Line> GetLinesInMap()
{
    ifstream fin("./bim_vector_map.txt");
    if (!fin)
    {
        cerr<<"please run in the directory included bim_vector_map.txt"<<endl;
        exit(0);
    }
    
    /*
     * Read axis vectors of poles in the map
     */
    std::vector<Line> lines;
    lines.reserve(poles_num_in_map);   // Set the capacity to (at least) the number of poles
    for ( int i = 0; i < poles_num_in_map; i++)
    {
        float data[6] = {0};  // Two points in one line, so it is 6 
        for ( int j = 0; j < 6; j++ )
            fin >> data[j];
        
        Eigen::Vector3d pts, pte;  // Start point and end point of the line
        pts << data[0], data[1], data[2];
        pte << data[3], data[4], data[5];
//         pts = pts / 1000; pte /= 1000; // from mm to m
        Line l(pts, pte); // axis vectors of poles
//         std::cout << i << ": "; l.PrintLine();
        lines.push_back(l);
    }
//     cout << '\n';
    fin.close();
    
    return lines;    
}

// angle unit:degree
double AngleBetweenTwoLines (Line &l1, Line &l2)
{
    Eigen::Vector3d v1, v2;
    v1 = l1.GetOrientation();
    v2 = l2.GetOrientation();

    double acos_numerator = abs( v1.dot(v2) );
    double acos_denominator_L = v1.norm();
    double acos_denominator_R = v2.norm();
    double acos_angle = acos_numerator / (acos_denominator_L * acos_denominator_R);
//     std::cout << "cos angle between two lines : " << cos_angle << std::endl;
    double cos_angle = std::acos(acos_angle) * 180 / M_PI; // rad to degree
    return cos_angle;
}

// returns the closest point to p on the line segment l 
// from: http://www.alecjacobson.com/weblog/?p=1486
Eigen::Vector3d PointProjectOntoSegment(Line &l, Eigen::Vector3d p)
{
    Eigen::Vector3d closest_point = Eigen::Vector3d::Zero();
    Eigen::Vector3d A = l.GetStartPoint();
    Eigen::Vector3d B = l.GetEndPoint();
    Eigen::Vector3d AB = l.GetOrientation();
    // squared distance from A to B
    double AB_squared = l.GetLength() * l.GetLength();
    // vector from A to p
    Eigen::Vector3d Ap = p - A;
    // Consider the line extending the segment, parameterized as A + t (B - A)
    // We find projection of point p onto the line. 
    // It falls where t = [(p-A) . (B-A)] / |B-A|^2
    double t = Ap.dot(AB)/AB_squared;
    if (t < 0.0) 
        // "Before" A on the line, just return A
        closest_point = A;
    else if (t > 1.0) 
        // "After" B on the line, just return B
        closest_point = B;
    else
        // projection lines "inbetween" A and B on the line
        closest_point = A + t * AB;
    return closest_point;
}

double EuclideanDistanceOfTwoLines(Line &l1, Line &l2)
{
    Eigen::Vector3d v1, v2;
    v1 = l1.GetOrientation();
    v2 = l2.GetOrientation();
    double dist = (v1-v2).norm();
    std::cout << "Distance between two lines : " << dist << std::endl;
    return dist;
}

double MidPointDistanceBetweenTwoLines(Line &l1, Line &l2)
{
    Eigen::Vector3d v1, v2;
    v1 = l1.GetMidPoint();
    v2 = l2.GetMidPoint();
    double dist = (v1-v2).norm();
//     std::cout << "Mid point distance between two lines : " << dist << std::endl;
    return dist;
}

// source lines will get the same orientation with target line
// void GetSameLineOrientationWithMap (std::vector<Line> &source_lines, std::vector<Line> &target_lines)
void GetSameLineOrientationWithMap (std::vector<Line> &source_lines, std::vector<Line> &target_lines, std::vector<Line> &reverse_lines)
{
    for (int n = 0; n < source_lines.size(); ++n)
    {
        Eigen::Vector3d target_pte = target_lines.at(n).GetEndPoint();
        Eigen::Vector3d source_pte = source_lines.at(n).GetEndPoint();
        Eigen::Vector3d source_pts = source_lines.at(n).GetStartPoint();
        
        double dist_pte_pte = Eigen::Vector3d(target_pte - source_pte).norm();
        double dist_pte_pts = Eigen::Vector3d(target_pte - source_pts).norm();
//         cout << "original orientation is: " << reverse_lines.at(n).GetOrientation().transpose() << endl;
        if (dist_pte_pte > dist_pte_pts)
        {
            reverse_lines.at(n).ReverseOrientation();
//             cout << "reverse orientation is: " << reverse_lines.at(n).GetOrientation().transpose() << endl;            
        }
    }
}

// find poles in map matched to those in camera
std::vector<Line> PoleIdMatch (std::vector<Line> &lines_in_cam)
{
    ParameterReader pd;
    
    int pole_id;
    std::vector<Line> line_matched;
    std::vector<Line> lines_in_map;
    line_matched.reserve(lines_in_cam.size());
    lines_in_map.reserve(poles_num_in_map);
    lines_in_map = GetLinesInMap();
    
    // 
    /* Get the smalleast distance and angle of line in map compared with the line in camera
     * cos_angle decreases in [0,pi], so can be compared by its value directly
     */
    
    for (int i = 0; i < lines_in_cam.size(); i++)
    {
        // 
        vector<double> lines_dist, lines_angle;
        lines_dist.reserve(lines_in_map.size());
        lines_angle.reserve(lines_in_map.size());
        
        double min_dist = 10;  // 10 meters
        const double min_cos_angle = atof(pd.getData("min_cos_angle").c_str()); // about cos 10° = 0.9848
        int min_dist_pole_id = 0, min_cos_angle_pole_id = 0;
//         cout << "[" << i << "] pole:" << endl;
        for (int j = 0; j < lines_in_map.size(); j++)
        {
            // if lines parallel
            double cos_angle = AngleBetweenTwoLines(lines_in_cam.at(i), lines_in_map.at(j));
//             std::cout << "[" << j << "]" << "cos_angle between two lines : " << cos_angle << std::endl;
            if ( cos_angle < min_cos_angle)  // means angle(degree) difference < 10°
            {
                double dist = MidPointDistanceBetweenTwoLines(lines_in_cam.at(i), lines_in_map.at(j));
                if (dist < min_dist)
                {
                    min_dist = dist;
                    min_dist_pole_id = j;
                }
            }
//             std::cout << std::endl;
        }
        pole_id = min_dist_pole_id;
        line_matched.push_back(lines_in_map.at(pole_id));
        cout << "pole_id [" << pole_id << "] :";
        lines_in_map.at(pole_id).PrintLine();
    }
    
    return line_matched;
}

/* Find the transform matrix from camera to map.
 * NOTE the base coordinate is map.
 * Refer to paper "A Linear Features-Constrained, Plücker Coordinates-Based, 
 * Closed-Form Registration Approach to Terrestrial LiDAR Point Clouds"
 */
Eigen::Matrix4d CoarseRegistration(std::vector<Line> &input_line, std::vector<Line> &target_line)
{
    const int n = input_line.size();
    std::vector<Eigen::Vector4d> ma_dot_stack;
    std::vector<Eigen::Vector4d> mb_dot_stack;
    std::vector<Eigen::Matrix4d> Q_la_dot_stack;
    std::vector<Eigen::Matrix4d> Q_ma_dot_stack;
    std::vector<Eigen::Matrix4d> W_lb_dot_stack;
    std::vector<Eigen::Matrix4d> W_prime_lb_dot_stack;
    
    ma_dot_stack.reserve(n);
    mb_dot_stack.reserve(n);
    Q_la_dot_stack.reserve(n);
    Q_ma_dot_stack.reserve(n);
    W_lb_dot_stack.reserve(n);
    W_prime_lb_dot_stack.reserve(n);
    
    Eigen::Matrix4d C_l  = Eigen::Matrix4d::Zero();
    for (int i = 0; i < n; i++)
    {
        // Use Plücker coordinate to represent line
        Eigen::Vector3d la = target_line.at(i).GetOrientation(); // for input vector and input_moment respective
        Eigen::Vector3d ma = target_line.at(i).GetMoment();
        Eigen::Vector3d lb = input_line.at(i).GetOrientation(); // for target_vector and target_moment respective
        Eigen::Vector3d mb = input_line.at(i).GetMoment();
//         std::cout << "before norm() la :\n" << la.transpose() << std::endl;
//         std::cout << "before norm() ma :\n" << ma.transpose() << std::endl;
//         std::cout << "before norm() lb :\n" << lb.transpose() << std::endl;
//         std::cout << "before norm() mb :\n" << mb.transpose() << std::endl;

        // Coordinate normalization
        // 求模太小容易出错
        double la_mod = la.norm(), lb_mod = lb.norm();
        la /= la_mod;
        ma /= la_mod;
        lb /= lb_mod;
        mb /= lb_mod;

        std::cout << "after norm() la :" << la.transpose() << std::endl;
        std::cout << "after norm() ma :" << ma.transpose() << std::endl;
        std::cout << "after norm() lb :" << lb.transpose() << std::endl;
        std::cout << "after norm() mb :" << mb.transpose() << std::endl;
        
        // expand to a 4*1 vector
        Eigen::Vector4d la_dot, ma_dot;  // _dot for quaternion
        Eigen::Vector4d lb_dot, mb_dot;
        la_dot << 0, la(0), la(1), la(2);
        ma_dot << 0, ma(0), ma(1), ma(2);
        lb_dot << 0, lb(0), lb(1), lb(2);
        mb_dot << 0, mb(0), mb(1), mb(2);
        ma_dot_stack.push_back(ma_dot);
        mb_dot_stack.push_back(mb_dot);
    
        Eigen::Matrix4d Q_la_dot;
        Q_la_dot << la_dot(0), -la_dot(1), -la_dot(2), -la_dot(3),
                    la_dot(1),  la_dot(0), -la_dot(3),  la_dot(2),
                    la_dot(2),  la_dot(3),  la_dot(0), -la_dot(1),
                    la_dot(3), -la_dot(2),  la_dot(1),  la_dot(0);
        Eigen::Matrix4d Q_ma_dot;
        Q_ma_dot << ma_dot(0), -ma_dot(1), -ma_dot(2), -ma_dot(3),
                    ma_dot(1),  ma_dot(0), -ma_dot(3),  ma_dot(2),
                    ma_dot(2),  ma_dot(3),  ma_dot(0), -ma_dot(1),
                    ma_dot(3), -ma_dot(2),  ma_dot(1),  ma_dot(0);
        Eigen::Matrix4d W_lb_dot;
        W_lb_dot << lb_dot(0), -lb_dot(1), -lb_dot(2), -lb_dot(3),
                    lb_dot(1),  lb_dot(0),  lb_dot(3), -lb_dot(2),
                    lb_dot(2), -lb_dot(3),  lb_dot(0),  lb_dot(1),
                    lb_dot(3),  lb_dot(2), -lb_dot(1),  lb_dot(0);
        Eigen::Matrix4d W_prime_lb_dot;
        W_prime_lb_dot << lb_dot(0),  lb_dot(1),  lb_dot(2),  lb_dot(3),
                          lb_dot(1),  lb_dot(0),  lb_dot(3), -lb_dot(2),
                          lb_dot(2), -lb_dot(3),  lb_dot(0),  lb_dot(1),
                          lb_dot(3),  lb_dot(2), -lb_dot(1),  lb_dot(0);
        Q_la_dot_stack.push_back(Q_la_dot);
        Q_ma_dot_stack.push_back(Q_ma_dot);
        W_lb_dot_stack.push_back(W_lb_dot);
        W_prime_lb_dot_stack.push_back(W_prime_lb_dot);

        C_l += Q_la_dot.transpose() * W_lb_dot;
        
//         std::cout << "in cycle [" << i << "] Q_la_dot :\n" << Q_la_dot << std::endl;
//         std::cout << "in cycle [" << i << "] Q_ma_dot :\n" << Q_ma_dot << std::endl;
//         std::cout << "in cycle [" << i << "] W_lb_dot :\n" << W_lb_dot << std::endl;
//         std::cout << "in cycle [" << i << "] C_l :\n" << C_l << std::endl;
        
    }
    C_l = -2 * C_l;

    // **************************** Get rotational matrix R *******************************
    /* 
     * Ａ 是一个４×４的实对称阵，它的所有特征值和特征向量都是实数并且不同特征值对应的特征向量正交
     * note: Eigen library dose not sort eigenvalues in any particular order
     * The eigenvalues are repeated according to their algebraic multiplicity,
     * so there are as many eigenvalues as rows in the matrix.
     * The eigenvalues are not sorted in any particular order.
     */
    Eigen::Matrix4d A = -(C_l + C_l.transpose()) / 2;
    Eigen::EigenSolver<Eigen::Matrix4d> es(A);    
    cout << "The eigenvalues of A are:" << endl << es.eigenvalues() << endl;
    cout << "The matrix of eigenvectors, V, is:" << endl << es.eigenvectors() << endl << endl;
    
    // find the maximum eigenvalue and its coresponding eigenvector
    std::complex<double> max_lambda = es.eigenvalues()[0];
    Eigen::Index max_lambda_index = 0;
    for (Eigen::Index i = 1; i < es.eigenvalues().rows(); i++)
    {
        std::complex<double> lambda = es.eigenvalues()[i];
        // print eigenvectors
//         std::cout << es.eigenvectors().col(i) << std::endl;
//         std::cout << es.eigenvectors() << std::endl;
        if (std::abs(max_lambda) < std::abs(lambda))  // compare their mod
        {
            max_lambda = lambda;
            max_lambda_index = i;
//             cout << "Mod of eigenvalues [" << i << "] is: " << std::abs(lambda) << endl;
//             cout << "max_lambda is: " << max_lambda << endl;
        }
    }
//     cout << "The maximum eigenvalues is" << es.eigenvalues()[max_lambda_index] << endl;
    Eigen::Vector4d r_dot = es.eigenvectors().col(max_lambda_index).real(); // get major eigen vector
    
    Eigen::Matrix4d W_r_dot;
    W_r_dot << r_dot(0), -r_dot(1), -r_dot(2), -r_dot(3),
               r_dot(1),  r_dot(0),  r_dot(3), -r_dot(2),
               r_dot(2), -r_dot(3),  r_dot(0),  r_dot(1),
               r_dot(3),  r_dot(2), -r_dot(1),  r_dot(0);
    Eigen::Matrix4d Q_r_dot;
    Q_r_dot << r_dot(0), -r_dot(1), -r_dot(2), -r_dot(3),
               r_dot(1),  r_dot(0), -r_dot(3),  r_dot(2),
               r_dot(2),  r_dot(3),  r_dot(0), -r_dot(1),
               r_dot(3), -r_dot(2),  r_dot(1),  r_dot(0);

    Eigen::Matrix4d R_dot = W_r_dot.transpose() * Q_r_dot;  // now R_dot is 4*4
//     cout << "R_dot is :\n" << R_dot << endl; 
    Eigen::Matrix3d R = R_dot.block(1,1,3,3);
//     cout << "rotation matrix is :\n" << R << endl; // final result of R is 3*3

    // Get translation vector t
    Eigen::Matrix4d C_m1 = Eigen::Matrix4d::Zero();
    Eigen::Matrix4d C_m2 = Eigen::Matrix4d::Zero();
    Eigen::Matrix4d C_m3 = Eigen::Matrix4d::Zero();
    Eigen::Matrix4d C_s  = Eigen::Matrix4d::Zero();
    for (int i = 0; i < n; i++)
    {
        
        Eigen::Vector4d ma_dot_prime = ma_dot_stack.at(i) - W_r_dot.transpose()*Q_r_dot*mb_dot_stack.at(i);
        Eigen::Matrix4d Q_ma_dot_prime;
        Q_ma_dot_prime << ma_dot_prime(0), -ma_dot_prime(1), -ma_dot_prime(2), -ma_dot_prime(3),
                          ma_dot_prime(1),  ma_dot_prime(0), -ma_dot_prime(3),  ma_dot_prime(2),
                          ma_dot_prime(2),  ma_dot_prime(3),  ma_dot_prime(0), -ma_dot_prime(1),
                          ma_dot_prime(3), -ma_dot_prime(2),  ma_dot_prime(1),  ma_dot_prime(0);
                    
        C_m1 += Eigen::Matrix4d::Identity();
        C_m2 += Q_ma_dot_prime.transpose() * W_lb_dot_stack.at(i);
        C_m3 += W_prime_lb_dot_stack.at(i).transpose()*Q_r_dot.transpose()*W_r_dot.transpose()*W_lb_dot_stack.at(i);
        
        // print C_m1, C_m2, C_m3 each time
//         std::cout << "in cycle [" << i << "]: C_m1 is:\n" << C_m1 << std::endl;
//         std::cout << "in cycle [" << i << "]: C_m2 is:\n" << C_m2 << std::endl;
//         std::cout << "in cycle [" << i << "]: C_m3 is:\n" << C_m3 << std::endl;
    }
    C_m1 =  2.0 * C_m1;
    C_m2 = -2.0 * C_m2;
    C_m3 =  2.0 * C_m3;
    C_s  = (C_m1 + C_m1.transpose() + C_m3 + C_m3.transpose()).inverse();
    
    double lambda2_numerator = r_dot.transpose()*C_s*(C_m2 + C_m2.transpose())*r_dot;
//     cout << lambda2_numerator << endl;;
    double lambda2_denominator = -1.0*r_dot.transpose()*C_s*r_dot;
    double lambda2 = lambda2_numerator / lambda2_denominator;
    Eigen::Vector4d s_dot = -1 * C_s * (C_m2 + C_m2.transpose())*r_dot-lambda2*C_s*r_dot;
//     Eigen::Vector4d r_dot_conjuate;
//     r_dot_conjuate <<  r_dot(0), -r_dot(1), -r_dot(2), -r_dot(3);  // r_dot is a quaternion
//     cout << "r_dot_conjuate is: " << r_dot_conjuate.transpose() << endl;
//     cout << "s_dot is: " << s_dot.transpose() << endl;
    
    /* Note the order of the arguments: the real w coefficient first,
     * while internally the coefficients are stored in the following order:
     * [x, y, z, w]
     */
    Eigen::Quaterniond r_dot_q (r_dot(0), r_dot(1), r_dot(2), r_dot(3) );
    Eigen::Quaterniond s_dot_q (s_dot(0), s_dot(1), s_dot(2), s_dot(3) );
//     cout << "r_dot is: " << r_dot.transpose() << endl; 
//     cout << "r_dot_q is: " << r_dot_q.w() << "\t"
//                            << r_dot_q.x() << "\t" 
//                            << r_dot_q.y() << "\t"
//                            << r_dot_q.z() << endl;
    
//     cout << "r_dot_q conjuate is: " << r_dot_q.conjugate().w() << "\t"
//                                     << r_dot_q.conjugate().x() << "\t"
//                                     << r_dot_q.conjugate().y() << "\t" 
//                                     << r_dot_q.conjugate().z() << endl;
//     std::cout << "out of cycle s_dot: " <<  s_dot.transpose()  << std::endl; // for test
//     cout << "s_dot_q is: " << s_dot_q.w() << "\t"
//                            << s_dot_q.x() << "\t" 
//                            << s_dot_q.y() << "\t"
//                            << s_dot_q.z() << endl;

    Eigen::Quaterniond t_dot_q = s_dot_q * r_dot_q.conjugate();
//     cout << "t_dot_q is: " << t_dot_q.w() << "\t"
//                            << t_dot_q.x() << "\t" 
//                            << t_dot_q.y() << "\t"
//                            << t_dot_q.z() << endl;
    Eigen::Vector3d t ;
    t << t_dot_q.x(), t_dot_q.y(), t_dot_q.z();
    t = 2*t;
//     cout << "translation is: " << t.transpose() << endl;

//     std::cout << "out of cycle C_l : \n" <<  C_l    << std::endl;
//     std::cout << "out of cycle C_m1 :\n" <<  C_m1   << std::endl;
//     std::cout << "out of cycle C_m2 :\n" <<  C_m2   << std::endl;
//     std::cout << "out of cycle C_m3 :\n" <<  C_m3   << std::endl;
//     std::cout << "out of cycle lambda2:" << lambda2 << std::endl;

    Eigen::Matrix4d coarse_registration_matrix;
    coarse_registration_matrix << R(0,0), R(0,1), R(0,2),  t(0),
                                  R(1,0), R(1,1), R(1,2),  t(1),
                                  R(2,0), R(2,1), R(2,2),  t(2), 
                                       0,      0,      0,     1;
//     q_w_curr = Eigen::Quaterniond (R.inverse()); // note the order of xyzw
    q_w_curr = Eigen::Quaterniond (R); // note the order of xyzw
    t_w_curr = t;
//     cout << "q_w_curr is: " << q_w_curr.w() << "\t"
//                             << q_w_curr.x() << "\t"
//                             << q_w_curr.y() << "\t"
//                             << q_w_curr.z() << endl;
    
    cout << "The CoarseRegistration matrix is: \n" << coarse_registration_matrix << endl;
    PrintEulerAnglesDegreeAndTranslation(coarse_registration_matrix);
//     cout << "The CoarseRegistration matrix(inversed) is: \n" << coarse_registration_matrix.inverse() << endl;
//     PrintEulerAnglesDegreeAndTranslation(coarse_registration_matrix.inverse());
//     cout << "After CoarseRegistration(inversed): \n" << initial_transform_matrix * coarse_registration_matrix.inverse() << endl;
//     PrintEulerAnglesDegreeAndTranslation(initial_transform_matrix * coarse_registration_matrix.inverse());
    
    return coarse_registration_matrix;
}

void BroadcastTf(Eigen::Quaterniond q_w_curr, Eigen::Vector3d t_w_curr )
{
    tf::Quaternion q;
    tf_transform.setOrigin(tf::Vector3(t_w_curr(0),
                                       t_w_curr(1),
                                       t_w_curr(2)));
    q.setW(q_w_curr.w());
    q.setX(q_w_curr.x());
    q.setY(q_w_curr.y());
    q.setZ(q_w_curr.z());
    tf_transform.setRotation(q);
}

Eigen::Matrix4d FineRegistration(std::vector<Line> &input_line, std::vector<Line> &target_line) 
{
//     std::cout << "Before ceres non linear optimized: " << endl;
//     std::cout << "q_w_curr is: " << q_w_curr.w() << "\t"
//                                  << q_w_curr.x() << "\t"
//                                  << q_w_curr.y() << "\t"
//                                  << q_w_curr.z() << "\n"
//               << "t_w_curr is: " << t_w_curr.transpose() << endl;

    Eigen::Matrix4d T_before_optimized = Eigen::Matrix4d::Identity();
    T_before_optimized.block(0,0,3,3) = q_w_curr.toRotationMatrix();
    T_before_optimized.block(0,3,3,1) = t_w_curr;
//     std::cout << "transform matrix is: " << "\n" << T_before_optimized << endl;
    std::cout << std::endl;

    //ceres::LossFunction *loss_function = NULL;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
    /* Ceres: version 1.12 for ceres::EigenQuaternionParameterization(),
     * so Eigen version can < 3.1 without being changed.
     */
    ceres::LocalParameterization *q_parameterization = new ceres::EigenQuaternionParameterization();
    ceres::Problem::Options problem_options;
    ceres::Problem problem(problem_options);
    problem.AddParameterBlock(parameters, 4, q_parameterization);
    problem.AddParameterBlock(parameters + 4, 3);
    
    const int n = input_line.size();
    for (int i = 0; i < n; i++)
    {
        // find the closest point on lines in map between midpoint of the lines in cam
        Eigen::Vector3d point_on_cam_line = input_line.at(i).GetMidPoint();
//         Eigen::Vector3d curr_point = point_on_cam_line;
        Eigen::Vector3d curr_point = q_w_curr.toRotationMatrix() * point_on_cam_line + t_w_curr;
        Eigen::Vector3d point_on_line = PointProjectOntoSegment(target_line.at(i), curr_point);
        Eigen::Vector3d unit_direction = target_line.at(i).GetOrientation() / target_line.at(i).GetLength();
        // 求垂足点
        Eigen::Vector3d point_a, point_b;
        point_a = 0.1 * unit_direction + point_on_line;
        point_b = -0.1 * unit_direction + point_on_line;

//         ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, point_a, point_b, 1.0);
        ceres::CostFunction *cost_function = LidarEdgeFactor::Create(point_on_cam_line, point_a, point_b, 1.0);
        problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);

//         std::cout << "in cycle [" << i << "]: curr_point is: " << curr_point.transpose() << std::endl;
//         std::cout << "in cycle [" << i << "]: point_on_line is: " << point_on_line.transpose() << std::endl;
//         std::cout << "in cycle [" << i << "]: unit_direction is: " << unit_direction.transpose() << std::endl;
//         std::cout << "in cycle [" << i << "]: point_a is: " << point_a.transpose() << std::endl;
//         std::cout << "in cycle [" << i << "]: point_b is: " << point_b.transpose() << std::endl;
//         std::cout << std::endl;
    }
    
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
//     options.max_num_iterations = 4;
    options.max_num_iterations = 4;
    options.minimizer_progress_to_stdout = false;
//     options.check_gradients = false;
    options.check_gradients = false;
    options.gradient_check_relative_precision = 1e-4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    
    std::cout << "After ceres non linear optimized: " << std::endl;
//     std::cout << "q_w_curr is: " << q_w_curr.w() << "\t"
//                                  << q_w_curr.x() << "\t"
//                                  << q_w_curr.y() << "\t"
//                                  << q_w_curr.z() << "\n"
//               << "t_w_curr is: " << t_w_curr.transpose() << endl;
    
//     Eigen::Matrix4d T_optimized = Eigen::Matrix4d::Identity();
//     T_optimized.block(0,0,3,3) = q_w_curr.toRotationMatrix();
//     T_optimized.block(0,3,3,1) = t_w_curr;
// 
//     Eigen::Matrix4d Final_transform = initial_transform_matrix * T_optimized.inverse();
//     Eigen::Matrix3d rot = Final_transform.block(0,0,3,3);
//     Eigen::Quaterniond q_w_cam = Eigen::Quaterniond(rot);
//     Eigen::Vector3d t_w_cam = Final_transform.block(0,3,3,1);
//     BroadcastTf(q_w_cam, t_w_cam);
    Eigen::Matrix4d Final_transform = Eigen::Matrix4d::Identity();
    Final_transform.block(0,0,3,3) = q_w_curr.toRotationMatrix();
    Final_transform.block(0,3,3,1) = t_w_curr;

    BroadcastTf(q_w_curr, t_w_curr);    
    std::cout << "Final transform matrix is: \n" << Final_transform << std::endl;
    PrintEulerAnglesDegreeAndTranslation(Final_transform);    
}

void TransformPointCloudPublish(pcl::PointCloud<pcl::PointXYZ>::Ptr source_pc, pcl::PointCloud<pcl::PointXYZ>::Ptr& transformed_pc, Eigen::Matrix4d& trans_matrix)
{
    pcl::transformPointCloud (*source_pc, *transformed_pc, trans_matrix);
    //Convert the transformed_pc to ROS message
    pcl::toROSMsg(*transformed_pc, transformed_pc_msg);
    transformed_pc_msg.header.frame_id = "world";
    transformed_pc_msg.header.stamp = ros::Time::now();
}

void CylinderLocalization_Callback(const sensor_msgs::PointCloud2ConstPtr& pc_msg)
{
    ROS_INFO("In callback!");
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg (*pc_msg, *cloud);

    std::vector<Line> poles_in_cam;
    CylinderRecognition(poles_in_cam, cloud);  // Get axis of poles in a depth camera
    
    std::vector<Line> poles_from_cam_to_map;
    poles_from_cam_to_map.reserve(poles_in_cam.size());
    LineAssociateToMap(poles_in_cam, poles_from_cam_to_map);  // Tranform to local coordinate
    
    std::vector<Line> pole_matched;
    pole_matched = PoleIdMatch(poles_from_cam_to_map); // Find those poles matched to the camera
//     GetSameLineOrientationWithMap(poles_in_cam, pole_matched); 
    GetSameLineOrientationWithMap(poles_from_cam_to_map, pole_matched, poles_in_cam); 
    
    Eigen::Matrix4d pose_estimated_from_coarse_registration;
    pose_estimated_from_coarse_registration = CoarseRegistration(poles_in_cam, pole_matched);
    
    Eigen::Matrix4d pose_estimated_final;
    pose_estimated_final = FineRegistration(poles_in_cam, pole_matched);
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_pc (new pcl::PointCloud<pcl::PointXYZ>);
    TransformPointCloudPublish(cloud, transformed_pc, pose_estimated_final);
}

int main (int argc, char** argv)
{   
    ros::init(argc, argv, "cylinders_localization");
    ros::NodeHandle nh;
    
    ros::Subscriber pc_sub = nh.subscribe("/camera/depth_registered/points", 1, CylinderLocalization_Callback);
    ros::Publisher pc_pub = nh.advertise<sensor_msgs::PointCloud2>("poles_pc", 1);
    
    double sample_rate = 1;
    ros::Rate naptime(sample_rate); // use to regulate loop rate 
    while (ros::ok())
    {        
        pc_pub.publish(transformed_pc_msg);

        static tf::TransformBroadcaster br;
        br.sendTransform(tf::StampedTransform(tf_transform, ros::Time::now(), "world", "camera_link"));
        
        ros::spinOnce(); //allow data update from callback; 
        naptime.sleep(); // wait for remainder of specified period; 
    }
    
    return 0;
}
