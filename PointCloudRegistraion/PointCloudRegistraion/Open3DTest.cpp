#include <iostream>

#include "DataType.h"
#include "Example.h"

#include <open3d/Open3D.h>

using namespace std;
using namespace open3d;

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
        auto pcd_down = pcd->VoxelDownSample(0.01);
        pcd_down->EstimateNormals(
            open3d::geometry::KDTreeSearchParamHybrid(0.1, 30));
        auto pcd_fpfh = pipelines::registration::ComputeFPFHFeature(
            *pcd_down, open3d::geometry::KDTreeSearchParamHybrid(0.25, 100));
        return std::make_tuple(pcd_down, pcd_fpfh);
    }

    std::tuple<std::shared_ptr<geometry::PointCloud>,
        std::shared_ptr<pipelines::registration::Feature>>
        PreprocessPointCloud(geometry::TriangleMesh mesh) {
        auto pcd = MeshToPointCloud(mesh, true);
        auto pcd_down = pcd->VoxelDownSample(0.01);
        pcd_down->EstimateNormals(
            open3d::geometry::KDTreeSearchParamHybrid(0.1, 30));
        auto pcd_fpfh = pipelines::registration::ComputeFPFHFeature(
            *pcd_down, open3d::geometry::KDTreeSearchParamHybrid(0.25, 100));
        return std::make_tuple(pcd_down, pcd_fpfh);
    }

    int Open3DTest()
    {
        RegistrationTestData testData = fail1;

        // input read meshes and pointclouds
        auto mesh_scene = make_shared<geometry::TriangleMesh>();
        io::ReadTriangleMesh(testData.scenePath, *mesh_scene);
        mesh_scene->ComputeVertexNormals();
        auto mesh_model = make_shared<geometry::TriangleMesh>();
        io::ReadTriangleMesh(testData.modelPath, *mesh_model);
        mesh_model->ComputeVertexNormals();

        shared_ptr<geometry::PointCloud> pc_scene, pc_model;
        shared_ptr<pipelines::registration::Feature> fpfh_scene, fpfh_model;
        tie(pc_scene, fpfh_scene) = PreprocessPointCloud(*mesh_scene);
        tie(pc_model, fpfh_model) = PreprocessPointCloud(*mesh_model);

        // global registration
        const string kMethodFeatureBased = "feature_based";
        const string kMethodCorres = "corres_based";

        const string method = kMethodCorres;

        bool mutual_filter = false;
        double maxCorrespondenceDistance = 0.075;
        int ransac_n = 3;

        auto correspondence_checker_edge_lenth = pipelines::registration::CorrespondenceCheckerBasedOnEdgeLength(0.9);
        auto correspondence_checker_distance = pipelines::registration::CorrespondenceCheckerBasedOnDistance(0.075);
        auto correspondence_checker_normal = pipelines::registration::CorrespondenceCheckerBasedOnNormal(0.52359878);
        auto convergence_criteria = pipelines::registration::RANSACConvergenceCriteria(100000, 0.999);
        auto estimate_pointtopoint = pipelines::registration::TransformationEstimationPointToPoint(false);

        pipelines::registration::RegistrationResult registration_result;

        vector <reference_wrapper<const pipelines::registration::CorrespondenceChecker>> correspondence_checker;
        correspondence_checker.push_back(correspondence_checker_edge_lenth);
        correspondence_checker.push_back(correspondence_checker_distance);
        correspondence_checker.push_back(correspondence_checker_normal);

        if (method == kMethodFeatureBased)
            registration_result = pipelines::registration::RegistrationRANSACBasedOnFeatureMatching(
                *pc_model, *pc_scene, *fpfh_scene, *fpfh_scene,
                mutual_filter, maxCorrespondenceDistance,
                estimate_pointtopoint, ransac_n, 
                correspondence_checker, convergence_criteria);

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

                registration_result = pipelines::registration::RegistrationRANSACBasedOnCorrespondence(
                    *pc_model, *pc_scene, mutual, maxCorrespondenceDistance,
                    estimate_pointtopoint, ransac_n,
                    correspondence_checker, convergence_criteria);
            }
            else {
                registration_result = pipelines::registration::RegistrationRANSACBasedOnCorrespondence(
                    *pc_model, *pc_scene, corres_ji, maxCorrespondenceDistance,
                    estimate_pointtopoint, ransac_n, 
                    correspondence_checker ,convergence_criteria);
            }
        }

        shared_ptr<geometry::TriangleMesh> mesh_result = open3d::io::CreateMeshFromFile(testData.modelPath);
        mesh_result->Transform(registration_result.transformation_);
        mesh_result->ComputeVertexNormals();

        // visualize result meshes and pointclouds

        auto color_scene = Eigen::Vector3d(1, 0, 0);
        auto color_model = Eigen::Vector3d(0, 1, 0);
        auto color_result = Eigen::Vector3d(0, 0, 1);

        mesh_scene->PaintUniformColor(color_scene);
        mesh_model->PaintUniformColor(color_model);
        mesh_result->PaintUniformColor(color_result);

        //visualization::DrawGeometries({ pc_scene, pc_model }, "PointCloud", 1600, 900);
        visualization::DrawGeometries({ mesh_scene, mesh_model, mesh_result }, "Mesh", 1600, 900);

        return 0;
    }
}