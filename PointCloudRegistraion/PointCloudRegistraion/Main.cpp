#include <iostream>

namespace Tester {
    int PointCloudSamplingTest();
    int ICPAfterPointCloudSamplingTest();
    int PCLTest();
    int Open3DTest();
}

using namespace Tester;
int main(int argc, char** argv)
{
    //return PointCloudSamplingTest();
    //return ICPAfterPointCloudSamplingTest();
    //return PCLTest();
    return Open3DTest();
}