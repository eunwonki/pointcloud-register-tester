#include <opencv2/core/mat.hpp>
#include <vector>
#include <set>

using namespace cv;
using namespace std;

namespace OpenCVWrapper
{
#pragma pack(push, 4)
    struct Vector3
    {
        float x;
        float y;
        float z;
    };

    typedef double Matx44dPtr[16];

    // *주의* swift 에서는 swift에서 MeshObj member 들을 allocate / delocate 해주어야한다.
    struct MeshObj
    {
        struct Vector3* vertices;
        struct Vector3* normals;
        unsigned int* faces;
        int numberOfMeshVertices;
        int numberOfMeshFaces;
        struct Vector3 meshOffset;

        // * 주의 *
        //  MeshObj struct 는 marshaling 때문에 vtable 을 가지면 안되므로 가상(virtual)멤버함수 (소멸자 포함)를 가질 수 없다.

        //~MeshObj();
        //void DeleteMembers();
    };


    MeshObj* LoadMeshObj(const char* objFilePath);
    Mat* MeshObj2MatPtr(const MeshObj* src);
    double Icp(const Mat model, const Mat scene, /*inout*/ Matx44dPtr pose
        , int iterations, float tolerence, float rejectionScale, int numLevels);

#pragma region Pointcloud
    long Vertices(Mat* pointcloud);
#pragma endregion

#pragma region pointcloud sampling
    Mat* GetSampledPointCloud(
        Mat* pointcloud,
        int maxNumNearFeaturePoint, vector<Vec3f> featurePoints,
        int outerSkipSize);

    Mat* GetDuplicateSampledFeaturePoint(
        Mat* pointcloud,
        int maxNumNearFeaturePoint, vector<Vec3f> featurePoints,
        int duplicateCount
    );

    set<int> GetKNearestPoint(Mat* pointcloud,
        vector<Vec3f> featurePoints, int numResult, int dimension);
    set<int> GetSamplingIndices(set<int> nearestIndices, int numInput, int outerSkipSize);
#pragma endregion
}
