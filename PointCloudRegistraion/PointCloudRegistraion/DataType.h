#pragma once

#ifndef DATATYPE_H
#define DATATYPE_H

#include <iostream>
#include <string>
#include <vector>
#include <opencv2/core/matx.hpp>

struct Vector3
{
    float x;
    float y;
    float z;
};

struct MeshObj
{
    struct Vector3* vertices;
    struct Vector3* normals;
    unsigned int* faces;
    int numberOfMeshVertices;
    int numberOfMeshFaces;
    struct Vector3 meshOffset;

    //~MeshObj();
    //void DeleteMembers();
};

typedef double Matx44dPtr[16];

struct RegistrationTestData 
{
    std::string name = "";
    std::string tag = "";

    std::string modelPath = "";
    std::string scenePath = "";

    double* initial = new double[16];

    std::vector<cv::Vec3f> featurePointsInScene;
    std::vector<cv::Vec3f> featurePointsInModel;
};

#endif