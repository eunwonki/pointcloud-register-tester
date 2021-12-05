#include <iostream>

namespace Tester {
    int PointCloudSamplingTest();
    int OpenCVTest();
    int PCLTest();
    int Open3DTest();
}

using namespace Tester;
int main(int argc, char** argv)
{
    //return PointCloudSamplingTest();
    //return OpenCVTest();
    //return PCLTest();
    return Open3DTest();
}