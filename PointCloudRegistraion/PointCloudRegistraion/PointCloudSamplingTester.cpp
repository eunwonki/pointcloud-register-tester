#include <iostream>
#include <opencv2/surface_matching/ppf_helpers.hpp> // to write ply

#include "OpenCVWrapper.h"

using namespace std;
using namespace cv;
using namespace OpenCVWrapper;

namespace Tester {
    int PointCloudSamplingTest()
    {
        static const char* PC_PATH = "model.obj";

        const vector<Vec3f> featurePoints = {
            Vec3f(0.113545f, 0.221563f, 0.183073f),
            Vec3f(0.144233f, 0.194138f, 0.211906f),
            Vec3f(0.072768f,0.191650f,0.210047f),
        };

        const int maxFeaturePoints = 300;
        const int outerSkipSize = 10;

        const int duplicateCount = 5;

        MeshObj* pcMesh = LoadMeshObj(PC_PATH);
        Mat* pc = MeshObj2MatPtr(pcMesh);
        cout << "original mesh vertices: " << Vertices(pc) << endl;
        cout << "num of featuer points: " << featurePoints.size() << endl;
        cout << "max collect num of nearby feature point: " << maxFeaturePoints << endl;
        
        cout << "outer skip size: " << outerSkipSize << endl;
        Mat* sampledPC = GetSampledPointCloud(pc, maxFeaturePoints, featurePoints, outerSkipSize);
        cout << "sampled point vertices (outer skip): " << Vertices(sampledPC) << endl;

        cv::ppf_match_3d::writePLY(*pc, "result\\original.ply");
        cv::ppf_match_3d::writePLY(*sampledPC, "result\\sampled_skip.ply");

        cout << "duplicate count near feature points: " << duplicateCount << endl;
        Mat* duplicateSampledPC = GetDuplicateSampledFeaturePoint(pc, maxFeaturePoints, featurePoints, duplicateCount);
        cout << "sampled point vertices (duplicate): " << Vertices(duplicateSampledPC) << endl;

        cv::ppf_match_3d::writePLY(*duplicateSampledPC, "result\\sample_duplicate.ply");

        delete duplicateSampledPC;
        delete sampledPC;
        delete pcMesh;
        delete pc;

        return 0;
    }
}
