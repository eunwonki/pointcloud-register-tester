#include <iostream>
#include <sstream>
#include <fstream>
#include <set>
#include <vector>
#include <string>

#include <opencv2/core/mat.hpp>
#include <opencv2/flann.hpp>
#include <opencv2/surface_matching.hpp>

#include "OpenCVWrapper.h"

using namespace std;
using namespace cv;

namespace OpenCVWrapper
{
    MeshObj* LoadMeshObj(const char* objFilePath) {
        if (objFilePath == nullptr || *objFilePath == '\0') return nullptr;

        ifstream fs(objFilePath);

        if (fs.good() == false) return nullptr;

        vector<Vector3> vertices;
        vector<Vector3> normals;

        string line;

        // check vertexcount to initial mat
        while (getline(fs, line))
        {
            istringstream iss(line);

            string property;

            iss >> property;

            if (property.find("#") == 0) continue;

            if (property == "v") {
                float x, y, z;
                iss >> x >> y >> z;

                vertices.push_back(Vector3{ x, y, z });
            }

            if (property == "vn")
            {
                float x, y, z;
                iss >> x >> y >> z;

                normals.push_back(Vector3{ x, y, z });
            }
        }

        MeshObj* mesh = new MeshObj();
        mesh->numberOfMeshVertices = (int)vertices.size();
        mesh->vertices = new Vector3[mesh->numberOfMeshVertices];  // vertices와 normals는 meshobj 반납할때 함께 반납.
        mesh->normals = new Vector3[mesh->numberOfMeshVertices];

        memcpy(mesh->vertices, vertices.data(), mesh->numberOfMeshVertices * sizeof(Vector3));
        memcpy(mesh->normals, normals.data(), mesh->numberOfMeshVertices * sizeof(Vector3));

        mesh->faces = nullptr; // Matching에 face 정보는 사용하지 않음...

        return mesh;
    }

    Mat* MeshObj2MatPtr(const MeshObj* src)
    {
        if (src == nullptr) return nullptr;

        const MeshObj& mesh = *reinterpret_cast<const MeshObj*>(src);

        // ref : Mat loadPLYSimple(const char* fileName, int withNormals)
        int size = mesh.numberOfMeshVertices;
        Vector3* n = mesh.normals;
        Vector3* v = mesh.vertices;

        bool withNormals = (v != nullptr);

        Mat* cloud = new Mat(size, withNormals ? 6 : 3, CV_32FC1);
        for (int i = 0; i < size; ++i)
        {
            float* data = cloud->ptr<float>(i);
            Vector3* data2 = reinterpret_cast<Vector3*>(data);
            *data2 = *v;
            if (withNormals)
            {
                double norm = sqrt(data[3] * data[3] + data[4] * data[4] + data[5] * data[5]);
                if (norm > 0.00001)
                {
                    data[3] /= static_cast<float>(norm);
                    data[4] /= static_cast<float>(norm);
                    data[5] /= static_cast<float>(norm);
                }
                ++data2;
                *data2 = *n;
                ++n;
            }
            ++v;
        }
        return cloud;
    }

    ppf_match_3d::Pose3DPtr Icp(const Mat model, const Mat scene, /*inout*/ Matx44dPtr pose
        , int iterations, float tolerence, float rejectionScale, int numLevels)
    {
        ppf_match_3d::ICP icp(iterations, tolerence, rejectionScale, numLevels, 0 /*sampleType*/, 1 /*numMaxCorr*/);
        vector<ppf_match_3d::Pose3DPtr> poses(1, new ppf_match_3d::Pose3D());
        auto& pose3d = poses.back();
        for (int i = 0; i < 16; ++i) pose3d->pose.val[i] = pose[i];
        icp.registerModelToScene(model, scene, poses);
        for (int i = 0; i < 16; ++i) pose[i] = (pose3d->pose.val[i]);
        return pose3d;
    }

#pragma region Pointcloud
    long Vertices(Mat* pointcloud)
    {
        return pointcloud->rows;
    }
#pragma endregion

#pragma region sampling pointcloud
    Mat* GetSampledPointCloudBySomeFeaturePoint(
        Mat* pointcloud,
        vector<Vec3f> featurePoints,
        float radius)
    {
        int dimension = 3;

        set<int> pointsInRangeIndices = GetPointsInRange(pointcloud, featurePoints, radius, dimension);

        auto sampledPC = new Mat(0, pointcloud->cols, pointcloud->type());

        for (int idx : pointsInRangeIndices)
        {
            Mat elem = pointcloud->row(idx);
            sampledPC->push_back(elem);
        }

        return sampledPC;
    }

    Mat* GetSampledPointCloud(
        Mat* pointcloud,
        int maxNumNearFeaturePoint, vector<Vec3f> featurePoints,
        int outerSkipSize)
    {
        Mat& pcMat = *reinterpret_cast<Mat*>(pointcloud);

        int numInput = pcMat.rows;
        int numResult = min(numInput, maxNumNearFeaturePoint);
        int dimension = 3;

        set<int> nearestIndices = GetKNearestPoints(pointcloud, featurePoints, numResult, dimension);
        set<int> samplingIndices = GetSamplingIndices(nearestIndices, numInput, outerSkipSize);

        auto sampledPC = new Mat(0, pcMat.cols, pcMat.type());

        for (int idx : samplingIndices)
        {
            Mat elem = pcMat.row(idx);
            sampledPC->push_back(elem);
        }

        return sampledPC;
    }

    Mat* GetDuplicateSampledFeaturePoint(
        Mat* pointcloud,
        int maxNumNearFeaturePoint, vector<Vec3f> featurePoints,
        int duplicateCount
    )
    {
        Mat& pcMat = *reinterpret_cast<Mat*>(pointcloud);

        int numInput = pcMat.rows;
        int numResult = min(numInput, maxNumNearFeaturePoint);
        int dimension = 3;

        set<int> nearestIndices = GetKNearestPoints(pointcloud, featurePoints, numResult, dimension);

        Mat* sampledPC = new Mat(pcMat);

        while (duplicateCount-- > 0)
        {
            for (int idx : nearestIndices)
            {
                Mat elem = pcMat.row(idx);
                sampledPC->push_back(elem);
            }
        }

        return sampledPC;
    }

    set<int> GetSamplingIndices(set<int> nearestIndices, int numInput, int outerSkipSize)
    {
        set<int> samplingIndices = nearestIndices;

        int cnt = 1;
        for (int i = 0; i < numInput; i++)
        {
            if (nearestIndices.find(i) != nearestIndices.end())
                continue;

            if (cnt >= outerSkipSize)
            {
                samplingIndices.insert(i);
                cnt = 1;
            }

            else
                cnt++;
        }

        return samplingIndices;
    }

    set<int> GetKNearestPoints(Mat* pointcloud,
        vector<Vec3f> featurePoints, int numResult, int dimension)
    {
        int numQuery = (int) featurePoints.size();

        Mat temp = pointcloud->clone().colRange(0, dimension); // remove normal data
        cv::flann::Index kdTree(temp, cv::flann::KDTreeIndexParams(8));

        Mat query = Mat(numQuery, dimension, temp.type(), featurePoints.data());
        Mat indices = Mat_<int>(numQuery, numResult);
        Mat dists = Mat_<float>(numQuery, numResult);

        kdTree.knnSearch(query, indices, dists, numResult);

        set<int> nearestPointIndices = set<int>();
        for (int i = 0; i < numQuery; i++) {
            for (int j = 0; j < numResult; j++) {
                int index = indices.at<int>(i, j);
                nearestPointIndices.insert(index);
            }
        }

        kdTree.release();
        query.release();
        indices.release();
        dists.release();
        temp.release();

        return nearestPointIndices;
    }

    set<int> GetPointsInRange(Mat* pointcloud,
        vector<Vec3f> featurePoints, float radius, int dimension)
    {
        size_t numQuery = featurePoints.size();
        int maxResult = 4096; // 성능을 위해 최대로 찾을 수 있는 점의 개수를 4096개로 제한.
        maxResult = min(pointcloud->rows, maxResult);
        int maxIndex = Vertices(pointcloud);

        Mat temp = pointcloud->clone().colRange(0, dimension); // remove normal data
        cv::flann::Index kdTree(temp, cv::flann::KDTreeIndexParams(1));

        set<int> pointsInRange = set<int>();
        for (int i = 0; i < numQuery; i++) {
            Mat query = Mat(1, dimension, temp.type(), &featurePoints[i]);

            vector<int> indices;
            vector<float> dists;

            int numResult = kdTree.radiusSearch(query, indices, dists,
                radius * radius, // radius search require square of radius
                maxResult, flann::SearchParams(4096));

            for (int j = 0; j < numResult; j++) {
                int index = indices[j];
                pointsInRange.insert(index);
            }

            query.release();
        }

        kdTree.release();
        temp.release();

        return pointsInRange;
    }
}