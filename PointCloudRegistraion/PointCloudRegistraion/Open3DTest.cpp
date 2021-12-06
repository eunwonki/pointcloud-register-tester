#include <iostream>

#include "DataType.h"
#include "Example.h"

#include <open3d/Open3D.h>

using namespace std;
using namespace open3d;

const float voxel_size = 0.005;
float radius_normal = voxel_size * 2;
float radius_feature = voxel_size * 5;

namespace Tester {
    shared_ptr<geometry::PointCloud> MeshToPointCloud(geometry::TriangleMesh mesh, bool withNormal) {
        auto pcd = make_shared <geometry::PointCloud>();

        pcd->points_.resize((int)mesh.vertices_.size());
        copy(mesh.vertices_.begin(), mesh.vertices_.end(), pcd->points_.begin());

        if (withNormal) {
            pcd->normals_.resize((int)mesh.vertex_normals_.size());
            copy(mesh.vertex_normals_.begin(), mesh.vertex_normals_.end(), pcd->normals_.begin());
        }

        return pcd;
    }

    std::tuple<std::shared_ptr<geometry::PointCloud>,
        std::shared_ptr<pipelines::registration::Feature>>
        PreprocessPointCloud(const char* file_name) {
        auto pcd = open3d::io::CreatePointCloudFromFile(file_name); // obj file은 지원하지 않음.
        auto pcd_down = pcd->VoxelDownSample(voxel_size);
        pcd_down->EstimateNormals(
            open3d::geometry::KDTreeSearchParamHybrid(radius_normal, 30));
        auto pcd_fpfh = pipelines::registration::ComputeFPFHFeature(
            *pcd_down, open3d::geometry::KDTreeSearchParamHybrid(radius_feature, 100));
        return std::make_tuple(pcd_down, pcd_fpfh);
    }

    std::tuple<std::shared_ptr<geometry::PointCloud>,

        std::shared_ptr<pipelines::registration::Feature>>
        PreprocessPointCloud(geometry::TriangleMesh mesh) {
        auto pcd = MeshToPointCloud(mesh, true );
        auto pcd_down = pcd->VoxelDownSample(voxel_size);
        pcd_down->EstimateNormals(
            open3d::geometry::KDTreeSearchParamHybrid(radius_normal, 30));
        auto pcd_fpfh = pipelines::registration::ComputeFPFHFeature(
            *pcd_down, open3d::geometry::KDTreeSearchParamHybrid(radius_feature, 100));
        return std::make_tuple(pcd_down, pcd_fpfh);
    }

    int Open3DTest()
    {
        //RegistrationTestData testData = fail1;
        PrintOpen3DVersion();

        // input read meshes
        auto mesh_scene = make_shared<geometry::TriangleMesh>();
        io::ReadTriangleMesh("data\\1\\scene.obj", *mesh_scene);
        mesh_scene->ComputeVertexNormals();
        auto mesh_model = make_shared<geometry::TriangleMesh>();
        io::ReadTriangleMesh("data\\1\\model.obj", *mesh_model);
        mesh_model->ComputeVertexNormals();

        // down sampling and build point cloud
        utility::Timer timer;
        //double down_sample = 0.005; //0.01;

        timer.Start();
        shared_ptr<geometry::PointCloud> pc_scene, pc_model;
        shared_ptr<pipelines::registration::Feature> fpfh_scene, fpfh_model;
        tie(pc_scene, fpfh_scene) = PreprocessPointCloud("data\\1\\target.pcd"); //(*mesh_scene);
            tie(pc_model, fpfh_model) = PreprocessPointCloud("data\\1\\source.pcd"); //(*mesh_model);
        timer.Stop();
        cout << "Sampling Time: " << timer.GetDuration() << " ms" << endl;

        // global registration
        const string kMethodFeatureBased = "feature_based";
        const string kMethodCorres = "corres_based";

        const string method = kMethodFeatureBased;

        bool mutual_filter = false;
        double maxCorrespondenceDistance = voxel_size * 1.5; //0.07;
        int ransac_n = 3;

        auto correspondence_checker_edge_lenth = pipelines::registration::CorrespondenceCheckerBasedOnEdgeLength(0.9);
        auto correspondence_checker_distance = pipelines::registration::CorrespondenceCheckerBasedOnDistance(maxCorrespondenceDistance); //0.075
        //auto correspondence_checker_normal = pipelines::registration::CorrespondenceCheckerBasedOnNormal(0.52359878);
        auto ransac_convergence_criteria = pipelines::registration::RANSACConvergenceCriteria(100000, 0.999);
        auto estimate_pointtopoint = pipelines::registration::TransformationEstimationPointToPoint(false);

        pipelines::registration::RegistrationResult global_registration_result;

        vector <reference_wrapper<const pipelines::registration::CorrespondenceChecker>> correspondence_checker;
        correspondence_checker.push_back(correspondence_checker_edge_lenth);
        correspondence_checker.push_back(correspondence_checker_distance);
        //correspondence_checker.push_back(correspondence_checker_normal);

        timer.Start();
        if (method == kMethodFeatureBased)
            global_registration_result = pipelines::registration::RegistrationRANSACBasedOnFeatureMatching(
                *pc_model, *pc_scene, *fpfh_scene, *fpfh_scene,
                mutual_filter, maxCorrespondenceDistance,
                estimate_pointtopoint, ransac_n,
                correspondence_checker, ransac_convergence_criteria);

        if (method == kMethodCorres)
        {
            int nPti = int(pc_model->points_.size());
            int nPtj = int(pc_scene->points_.size());

            geometry::KDTreeFlann feature_tree_i(*fpfh_model);
            geometry::KDTreeFlann feature_tree_j(*fpfh_scene);

            pipelines::registration::CorrespondenceSet corres_ji;
            vector<int> i_to_j(nPti, -1);

            // Buffer all correspondences
            for (int j = 0; j < nPtj; j++) {
                std::vector<int> corres_tmp(1);
                std::vector<double> dist_tmp(1);

                feature_tree_i.SearchKNN(Eigen::VectorXd(fpfh_scene->data_.col(j)),
                    1, corres_tmp, dist_tmp);
                int i = corres_tmp[0];
                corres_ji.push_back(Eigen::Vector2i(i, j));
            }

            pipelines::registration::CorrespondenceSet mutual;
            if (mutual_filter) {
                pipelines::registration::CorrespondenceSet mutual;
                for (auto& corres : corres_ji) {
                    int j = corres(1);
                    int j2i = corres(0);

                    std::vector<int> corres_tmp(1);
                    std::vector<double> dist_tmp(1);
                    feature_tree_j.SearchKNN(
                        Eigen::VectorXd(fpfh_model->data_.col(j2i)), 1,
                        corres_tmp, dist_tmp);
                    int i2j = corres_tmp[0];
                    if (i2j == j) {
                        mutual.push_back(corres);
                    }
                }

                global_registration_result = pipelines::registration::RegistrationRANSACBasedOnCorrespondence(
                    *pc_model, *pc_scene, mutual, maxCorrespondenceDistance,
                    estimate_pointtopoint, ransac_n,
                    correspondence_checker, ransac_convergence_criteria);
            }
            else {
                global_registration_result = pipelines::registration::RegistrationRANSACBasedOnCorrespondence(
                    *pc_model, *pc_scene, corres_ji, maxCorrespondenceDistance,
                    estimate_pointtopoint, ransac_n,
                    correspondence_checker, ransac_convergence_criteria);
            }
        }
        timer.Stop();
        cout << "Global Registration Time: " << timer.GetDuration() << " ms" << endl;

        shared_ptr<geometry::TriangleMesh> mesh_global_result = open3d::io::CreateMeshFromFile("data\\1\\model.obj");
        mesh_global_result->Transform(global_registration_result.transformation_);
        mesh_global_result->ComputeVertexNormals();

        // local registration using g-icp
        int iterations = 100;
        auto icp_convergence_criteria = pipelines::registration::ICPConvergenceCriteria(1e-6, 1e-6, iterations);

        timer.Start();
        auto local_registration_result = pipelines::registration::RegistrationICP(
            *pc_model, *pc_scene, maxCorrespondenceDistance, global_registration_result.transformation_,
            pipelines::registration::TransformationEstimationPointToPlane(),
            icp_convergence_criteria);
        timer.Stop();
        cout << "Local Registration Time: " << timer.GetDuration() << " ms" << endl;

        shared_ptr<geometry::TriangleMesh> mesh_local_result = open3d::io::CreateMeshFromFile("data\\1\\model.obj");
        mesh_local_result->Transform(local_registration_result.transformation_);
        mesh_local_result->ComputeVertexNormals();

        // visualize result meshes and pointclouds

        auto color_scene = Eigen::Vector3d(1, 0, 0);
        auto color_model = Eigen::Vector3d(0, 1, 0);
        auto color_global_result = Eigen::Vector3d(0, 0, 1);
        auto color_local_result = Eigen::Vector3d(1, 1, 0);

        mesh_scene->PaintUniformColor(color_scene);
        mesh_model->PaintUniformColor(color_model);
        mesh_global_result->PaintUniformColor(color_global_result);
        mesh_local_result->PaintUniformColor(color_local_result);

        //visualization::DrawGeometries({ pc_scene, pc_model }, "PointCloud", 1600, 900);
        visualization::DrawGeometries({
            mesh_scene,
            mesh_model,
            mesh_global_result,
            mesh_local_result,
            }, "Mesh", 1600, 900);

        return 0;
    }
}