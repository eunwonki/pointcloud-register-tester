#include <iostream>
#include "OpenCVWrapper.h"
#include "Example.h"
#include <opencv2/surface_matching/ppf_helpers.hpp>
#include <pcl/console/time.h>
#include <boolinq/boolinq.h>

using namespace OpenCVWrapper;
using namespace std;
using namespace cv::ppf_match_3d;
using namespace boolinq;

namespace Tester {
    int OpenCVTest()
    {
        // Parameters
        int iterations = 100;
        float rejectionScale = 1.5f;
        float tolerence = 0.005f;
        int numLevels = 6;
        const float samplingRange = 0.05f;

        RegistrationTestData testData = fieldtest5;

        cout << testData.name << endl;
        Mat* model = MeshObj2MatPtr(LoadMeshObj(testData.modelPath.c_str()));
        Mat* scene = MeshObj2MatPtr(LoadMeshObj(testData.scenePath.c_str()));

        auto pointsInScene = from(testData.featurePoints).select([](const FeaturePoint& point) { return point.pointInScene; }).toStdVector();
        auto pointsInModel = from(testData.featurePoints).select([](const FeaturePoint& point) { return point.pointInModel; }).toStdVector();

        pcl::console::TicToc time;
        time.tic();

        Mat* sampledScene = GetSampledPointCloudBySomeFeaturePoint(scene, pointsInScene, samplingRange);
        Mat* sampledModel = GetSampledPointCloudBySomeFeaturePoint(model, pointsInModel, samplingRange);

        cout << "sampling time: " << time.toc() << " ms" << endl;

        writePLY(*reinterpret_cast<Mat*>(scene), "result\\original_scene.ply");
        writePLY(*reinterpret_cast<Mat*>(sampledScene), "result\\sampled_scene.ply");
        writePLY(*reinterpret_cast<Mat*>(model), "result\\original_model.ply");
        writePLY(*reinterpret_cast<Mat*>(sampledModel), "result\\sampled_model.ply");

        Matx44d initialpose;
        auto pose = new double[16];
        memcpy(pose, testData.initial, 16 * sizeof(double));
        initialpose = Matx44d(pose);

        Mat icp_in = transformPCPose(*model, initialpose);
        writePLY(icp_in, "result\\icp_in.ply");

        cout << "initial pose = " << endl;
        for (int i = 0; i < 16; i++)
            cout << pose[i] << endl;

        time.tic();
        auto result = Icp(*sampledModel, *sampledScene, pose, iterations, tolerence, rejectionScale, numLevels);

        cout << "icp time: " << time.toc() << " ms" << endl;
        cout << "residual = " << result->residual << endl;
        cout << "After Sampling result pose = " << endl;
        for (int i = 0; i < 16; i++)
            cout << pose[i] << endl;

        Mat icp_out = transformPCPose(*model, result->pose);
        writePLY(icp_out, "result\\icp_out.ply");

        return 0;
    }
}