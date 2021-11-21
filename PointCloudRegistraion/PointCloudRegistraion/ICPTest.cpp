#include <iostream>
#include "OpenCVWrapper.h"

using namespace OpenCVWrapper;
using namespace std;

namespace Tester {
    int ICPAfterPointCloudSamplingTest()
    {
        // Parameters
        int iterations = 100;
        float rejectionScale = 1.5f;
        float tolerence = 0.005f;
        int numLevels = 6;

        const int maxFeaturePointsInScene = 300;
        const int outerSkipSizeInScene = 10;
        const vector<Vec3f> featurePointsInScene = {
            Vec3f(0.21589498f, -0.177592f, -0.2191054f),
        };

        const int maxFeaturePointsInModel = 500;
        const int outerSkipSizeInModel = 10;
        const vector<Vec3f> featurePointsInModel = {
            Vec3f(0.113545f, 0.221563f, 0.183073f),
            //Vec3f(0.144233f, 0.194138f, 0.211906f),
            //Vec3f(0.072768f, 0.191650f, 0.210047f),
        };

        static const char* MODEL_PATH = "model.obj";
        static const char* SCENE_PATH = "scene.obj";

        Matx44dPtr pose = {
            -0.21549509465694427, 
            -0.51570183038711548,
            0.82922476530075073,
            0.20281513035297394,
            -0.0032726526260375977,
            0.84955275058746338,
            0.52749329805374146,
            -0.46202057600021362,
            -0.97649949789047241,
            0.11095847189426422,
            -0.18476219475269318,
            -0.09898819774389267,
            0,
            0,
            0,
            1
        };

        Mat* model = MeshObj2MatPtr(LoadMeshObj(MODEL_PATH));
        Mat* scene = MeshObj2MatPtr(LoadMeshObj(SCENE_PATH));

        Mat* sampledScene = GetSampledPointCloud(scene, maxFeaturePointsInScene, featurePointsInScene, outerSkipSizeInScene);
        Mat* sampledModel = GetSampledPointCloud(model, maxFeaturePointsInModel, featurePointsInModel, outerSkipSizeInModel);

        cout << "initial pose = " << endl;
        for (int i = 0; i < 16; i++)
            cout << pose[i] << endl;

        double residual = Icp(*sampledModel, *sampledScene, pose, iterations, tolerence, rejectionScale, numLevels);

        cout << "residual = " << residual << endl;

        cout << "result pose = " << endl;
        for (int i = 0; i < 16; i++)
            cout << pose[i] << endl;

        return 0;
    }
}