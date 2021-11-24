#include <iostream>
#include "OpenCVWrapper.h"
#include "Example.h"
#include <opencv2/surface_matching/ppf_helpers.hpp>

using namespace OpenCVWrapper;
using namespace std;
using namespace cv::ppf_match_3d;

namespace Tester {
    int ICPAfterPointCloudSamplingTest()
    {
        // Parameters
        int iterations = 100;
        float rejectionScale = 1.5f;
        float tolerence = 0.005f;
        int numLevels = 6;
        const float samplingRange = 0.05f;

        RegistrationTestData testData = fail2;

        cout << testData.name << endl;
        Mat* model = MeshObj2MatPtr(LoadMeshObj(testData.modelPath.c_str()));
        Mat* scene = MeshObj2MatPtr(LoadMeshObj(testData.scenePath.c_str()));

        Mat* sampledScene = GetSampledPointCloudBySomeFeaturePoint(scene, testData.featurePointsInScene, samplingRange);
        Mat* sampledModel = GetSampledPointCloudBySomeFeaturePoint(model, testData.featurePointsInModel, samplingRange);

        writePLY(*reinterpret_cast<Mat*>(scene), "result\\original_scene.ply");
        writePLY(*reinterpret_cast<Mat*>(sampledScene), "result\\sampled_scene.ply");
        writePLY(*reinterpret_cast<Mat*>(model), "result\\original_model.ply");
        writePLY(*reinterpret_cast<Mat*>(sampledModel), "result\\sampled_model.ply");

        auto pose = testData.initial;
        auto initialpose = Matx44d(pose);
        cout << "initial pose = " << endl;
        for (int i = 0; i < 16; i++)
            cout << pose[i] << endl;

        Mat icp_in = transformPCPose(*model, initialpose);
        writePLY(icp_in, "result\\icp_in.ply");

        auto result = Icp(*sampledModel, *sampledScene, pose, iterations, tolerence, rejectionScale, numLevels);

        cout << "residual = " << result -> residual << endl;

        cout << "result pose = " << endl;
        for (int i = 0; i < 16; i++)
            cout << pose[i] << endl;

        Mat icp_out = transformPCPose(*model, result -> pose);
        writePLY(icp_out, "result\\icp_out.ply");

        return 0;
    }
}