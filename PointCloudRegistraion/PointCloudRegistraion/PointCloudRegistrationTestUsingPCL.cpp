#include <iostream>

#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>
#include <pcl/console/time.h>
#include <pcl/io/obj_io.h>
#include <pcl/io/ply_io.h>
#include "DataType.h"
#include "Example.h"

using namespace pcl;
using namespace std;

// PCL에서 바꿔가며 확인해볼 수 있는 것들에 대한 확인.
// 목표: Local Minima를 피하면서 빠르게 돌아갈 수 있는 알고리즘을 찾는 것.

// TODO PCL Visulize 연구하는 환경개선.
// TODO ICP 설정 바꿔가며 확인.

namespace Tester {
    void print4x4Matrix(const Eigen::Matrix4d& matrix)
    {
        printf("matrix to pointcloud:\n");
        
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                printf("%6.6f\n", matrix(i, j));

        cout << endl;
    }

    Eigen::Matrix4d doubleArrayToMatrix(double* array)
    {
        Eigen::Matrix4d matrix;
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                matrix(i, j) = array[i * 4 + j];
        return matrix;
    }

    int PointCloudRegistrationTestUsingPCL()
    {
        RegistrationTestData testData = fail1;

        int iterations = 200;
        double transformationEpsilon = 1e-8;
        double maxCorrespondenceDistance = 1.0;

        cout << testData.tag << "test" << endl;

        PointCloud<PointXYZ>::Ptr cloud_model(new PointCloud<PointXYZ>);
        PointCloud<PointXYZ>::Ptr cloud_scene(new PointCloud<PointXYZ>);
        PointCloud<PointXYZ>::Ptr cloud_in(new PointCloud<PointXYZ>);

        console::TicToc time;
        time.tic();

        io::loadOBJFile(testData.modelPath, *cloud_model);
        io::loadOBJFile(testData.scenePath, *cloud_scene);

        cout << "initial pose" << endl;
        Eigen::Matrix4d in_matrix = doubleArrayToMatrix(testData.initial);
        print4x4Matrix(in_matrix);

        transformPointCloud(*cloud_model, *cloud_in, in_matrix);

        io::savePLYFile("result\\scene.ply", *cloud_scene);
        io::savePLYFile("result\\icp_in.ply", *cloud_in);
        
        GeneralizedIterativeClosestPoint<PointXYZ, PointXYZ> icp;
        icp.setMaximumIterations(iterations);
        icp.setTransformationEpsilon(transformationEpsilon);
        icp.setMaxCorrespondenceDistance(maxCorrespondenceDistance);
        icp.setInputSource(cloud_model);
        icp.setInputTarget(cloud_scene);
        icp.align(*cloud_model, in_matrix.matrix());
        cout << "Applied " << iterations << " ICP iteration(s) in " << time.toc() << " ms" << endl;

        if (icp.hasConverged())
        {
            double efe = icp.getEuclideanFitnessEpsilon();
            double score = icp.getFitnessScore();
            auto matrix = icp.getFinalTransformation().cast<double>();

            cout << "euclidean fitness epsilon: " << efe << endl;
            cout << "score: " << score << endl;
            print4x4Matrix(matrix);
        }
        else
        {
            cout << "icp did not converge...";
        }

        return 0;
    }
}