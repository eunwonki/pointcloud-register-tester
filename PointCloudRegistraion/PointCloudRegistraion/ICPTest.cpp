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

        RegistrationTestData testData = fail1;

        const int maxFeaturePointsInScene = 1000;
        const int outerSkipSizeInScene = 1000;
        const vector<Vec3f> featurePointsInScene = {
            Vec3f(0.21589498f, -0.177592f, -0.2191054f),
        };

        const int maxFeaturePointsInModel = 3000;
        const int outerSkipSizeInModel = 1000;
        const vector<Vec3f> featurePointsInModel = {
            Vec3f(0.113545f, 0.221563f, 0.183073f),
            //Vec3f(0.144233f, 0.194138f, 0.211906f),
            //Vec3f(0.072768f, 0.191650f, 0.210047f),
        };

        Mat* model = MeshObj2MatPtr(LoadMeshObj(testData.modelPath.c_str()));
        Mat* scene = MeshObj2MatPtr(LoadMeshObj(testData.scenePath.c_str()));

        Mat* sampledScene = GetSampledPointCloud(scene, maxFeaturePointsInScene, featurePointsInScene, outerSkipSizeInScene);
        Mat* sampledModel = GetSampledPointCloud(model, maxFeaturePointsInModel, featurePointsInModel, outerSkipSizeInModel);

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