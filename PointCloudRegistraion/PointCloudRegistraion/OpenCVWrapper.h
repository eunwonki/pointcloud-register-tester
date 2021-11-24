#pragma once

#ifndef OPENCVWRAPPER_H
#define OPENCVWRAPPER_H

#include <opencv2/core/mat.hpp>
#include <opencv2/surface_matching.hpp>
#include <vector>
#include <set>
#include "DataType.h"

using namespace cv;
using namespace std;

namespace OpenCVWrapper
{
    MeshObj* LoadMeshObj(const char* objFilePath);
    Mat* MeshObj2MatPtr(const MeshObj* src);
    ppf_match_3d::Pose3DPtr Icp(const Mat model, const Mat scene, /*inout*/ Matx44dPtr pose
        , int iterations, float tolerence, float rejectionScale, int numLevels);

#pragma region Pointcloud
    long Vertices(Mat* pointcloud);
#pragma endregion

#pragma region pointcloud sampling
    Mat* GetSampledPointCloudBySomeFeaturePoint(
        Mat* pointcloud,
        vector<Vec3f> featurePoints,
        float radius);

    Mat* GetSampledPointCloud(
        Mat* pointcloud,
        int maxNumNearFeaturePoint, vector<Vec3f> featurePoints,
        int outerSkipSize);

    Mat* GetDuplicateSampledFeaturePoint(
        Mat* pointcloud,
        int maxNumNearFeaturePoint, vector<Vec3f> featurePoints,
        int duplicateCount
    );

    set<int> GetKNearestPoints(Mat* pointcloud,
        vector<Vec3f> featurePoints, int numResult, int dimension);
    set<int> GetPointsInRange(Mat* pointcloud,
        vector<Vec3f> featurePoints, float radius, int dimension);
    set<int> GetSamplingIndices(set<int> nearestIndices, int numInput, int outerSkipSize);
#pragma endregion
}

#endif
