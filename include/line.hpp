#ifndef _LINE_H
#define _LINE_H

#include<iostream>
#include <fstream>
#include<cmath>
using namespace std;

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

class Point
{
public:
    Point() : x_(0), y_(0), z_(0), pt_( 0, 0, 0) {}; 
    // pt_ 不是赋值，是初始化
    Point(double x, double y, double z) : x_(x), y_(y), z_(z), pt_( x, y, z) {};
    Point(Eigen::Vector3d pt) : pt_(pt) {};
    
    Eigen::Vector3d GetPoint();
    void SetPoint(double x, double y, double z);
    void SetPoint(Eigen::Vector3d pt);
    void PrintPoint();
        
private:
    double x_, y_, z_;
    Eigen::Vector3d pt_;
};

Eigen::Vector3d Point::GetPoint()
{
    return pt_;
}

void Point::SetPoint(double x, double y, double z)
{
    pt_ << x, y, z;
}

void Point::SetPoint(Eigen::Vector3d pt)
{
    pt_ = pt;
}

void Point::PrintPoint()
{
    cout << "Point: ("<< pt_.transpose() << ")" << endl;
}

class Line: public Point   
{
public:
    Line() : pts_(), pte_() {};
    // can be initialized by Eigen::Vector3d from the construction function of class Point?
    Line(Point pts, Point pte) : pts_(pts), pte_(pte){};
    Line(Eigen::Vector3d pts, Eigen::Vector3d pte) : pts_(pts), pte_(pte){};
    Eigen::Vector3d GetStartPoint();
    Eigen::Vector3d GetEndPoint();
    Eigen::Vector3d GetMidPoint();
    Eigen::Vector3d GetOrientation();
    Eigen::Vector3d GetMoment();

    double GetLength();
    void PrintLenth();
    void PrintLine();   // print two end points and the length of the line
    void SetPoints(const Eigen::Vector3d &pts, const Eigen::Vector3d &pte);
    void ReverseOrientation(); // swap the pts_ and pte_
    
private:
    class Point pts_, pte_;
};

Eigen::Vector3d Line::GetStartPoint()
{
    return Eigen::Vector3d( pts_.GetPoint() );
}

Eigen::Vector3d Line::GetEndPoint()
{
    return Eigen::Vector3d( pte_.GetPoint() );
}

Eigen::Vector3d Line::GetMidPoint()
{
    return Eigen::Vector3d( ( pts_.GetPoint()+pte_.GetPoint() )/2 );
}


Eigen::Vector3d Line::GetOrientation()
{
    return Eigen::Vector3d(pte_.GetPoint()-pts_.GetPoint());
}

Eigen::Vector3d Line::GetMoment()
{
    return Eigen::Vector3d( pte_.GetPoint().cross(pts_.GetPoint()) );
}


double Line::GetLength()
{
    return Eigen::Vector3d(pte_.GetPoint()-pts_.GetPoint()).norm();
}

void Line::PrintLenth()
{
    cout << "Length: " << Eigen::Vector3d(pte_.GetPoint()-pts_.GetPoint()).norm() << endl;
}

void Line::PrintLine()
{
    cout << "Line: < ("<< pts_.GetPoint().transpose() << "), " 
        << "("<< pte_.GetPoint().transpose() << "), "
        << GetLength() << " >" << endl;
}

void Line::SetPoints(const Eigen::Vector3d &pts, const Eigen::Vector3d &pte)
{
    pts_.SetPoint(pts);
    pte_.SetPoint(pte);
}

void Line::ReverseOrientation()
{
    Eigen::Vector3d temp_start_point, temp_end_point;
    temp_start_point = GetStartPoint();
    temp_end_point = GetEndPoint();
    pts_.SetPoint(temp_end_point);
    pte_.SetPoint(temp_start_point);
}

// from: laserMapping.h amd laserMapping.cpp in A-LOAM
// q_last_curr is the quaternion based on the last frame
// t_last_curr is the translation based on the last frame
struct LidarEdgeFactor
{
    LidarEdgeFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_a_,
                    Eigen::Vector3d last_point_b_, double s_)
        : curr_point(curr_point_), last_point_a(last_point_a_), last_point_b(last_point_b_), s(s_) {}

    template <typename T>
    bool operator()(const T *q, const T *t, T *residual) const
    {

        Eigen::Matrix<T, 3, 1> cp (T(curr_point.x()), T(curr_point.y()), T(curr_point.z()));
        Eigen::Matrix<T, 3, 1> lpa (T(last_point_a.x()), T(last_point_a.y()), T(last_point_a.z()));
        Eigen::Matrix<T, 3, 1> lpb (T(last_point_b.x()), T(last_point_b.y()), T(last_point_b.z()));

        Eigen::Quaternion<T> q_last_curr (q[3], q[0], q[1], q[2]);
        Eigen::Quaternion<T> q_identity(T(1), T(0), T(0), T(0));
        q_last_curr = q_identity.slerp(T(s), q_last_curr);
        Eigen::Matrix<T, 3, 1> t_last_curr(T(s) * t[0], T(s) * t[1], T(s) * t[2]);

        Eigen::Matrix<T, 3, 1> lp;
        lp = q_last_curr * cp + t_last_curr;

        Eigen::Matrix<T, 3, 1> nu = (lp - lpa).cross(lp - lpb);
        Eigen::Matrix<T, 3, 1> de = lpa - lpb;

        residual[0] = nu.x() / de.norm();
        residual[1] = nu.y() / de.norm();
        residual[2] = nu.z() / de.norm();

        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_a_,
                                       const Eigen::Vector3d last_point_b_, const double s_)
    {
        return (new ceres::AutoDiffCostFunction<
                LidarEdgeFactor, 3, 4, 3>(
            new LidarEdgeFactor(curr_point_, last_point_a_, last_point_b_, s_)));
    }

    Eigen::Vector3d curr_point, last_point_a, last_point_b;
    double s;
};

# endif