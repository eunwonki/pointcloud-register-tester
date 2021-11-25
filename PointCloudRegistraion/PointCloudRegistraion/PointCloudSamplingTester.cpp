#include <iostream>
#include <opencv2/surface_matching/ppf_helpers.hpp> // to write ply

#include "OpenCVWrapper.h"
#include "Example.h"

#include <boolinq/boolinq.h>

using namespace std;
using namespace cv;
using namespace OpenCVWrapper;

namespace Tester {
    int PointCloudSamplingTest()
    {
        RegistrationTestData testData = success2;

        const int maxFeaturePoints = 300;
        const int outerSkipSize = 10;

        const int duplicateCount = 5;

        const float searchRange = 0.05f;

        auto featurePoints = boolinq::from(testData.featurePoints).select([](const FeaturePoint& point) { return point.pointInScene; }).toStdVector();

        MeshObj* pcMesh = LoadMeshObj(testData.scenePath.c_str());
        Mat* pc = MeshObj2MatPtr(pcMesh);
        cout << "original mesh vertices: " << Vertices(pc) << endl;
        cout << "num of featuer points: " << featurePoints.size() << endl;
        cout << "max collect num of nearby feature point: " << maxFeaturePoints << endl;
        
        cout << "outer skip size: " << outerSkipSize << endl;
        Mat* sampledPC = GetSampledPointCloud(pc, maxFeaturePoints, featurePoints, outerSkipSize);
        cout << "sampled point vertices (outer skip): " << Vertices(sampledPC) << endl;

        cv::ppf_match_3d::writePLY(*pc, "result\\original.ply");
        cv::ppf_match_3d::writePLY(*sampledPC, "result\\sampled_skip.ply");

        cout << "\nduplicate count near feature points: " << duplicateCount << endl;
        Mat* duplicateSampledPC = GetDuplicateSampledFeaturePoint(pc, maxFeaturePoints, featurePoints, duplicateCount);
        cout << "sampled point vertices (duplicate): " << Vertices(duplicateSampledPC) << endl;

        cv::ppf_match_3d::writePLY(*duplicateSampledPC, "result\\sample_duplicate.ply");

        cout << "\nradius range: " << searchRange << endl;
        Mat* rangeSampledPC = GetSampledPointCloudBySomeFeaturePoint(pc, featurePoints, searchRange);
        cout << "sampled point vertices (radius range): " << Vertices(rangeSampledPC) << endl;

        cv::ppf_match_3d::writePLY(*rangeSampledPC, "result\\sample_range.ply");

        delete rangeSampledPC;
        delete duplicateSampledPC;
        delete sampledPC;
        delete pcMesh;
        delete pc;

        return 0;
    }
}
