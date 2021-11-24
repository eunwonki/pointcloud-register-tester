#include <iostream>

namespace Tester {
    int PointCloudSamplingTest();
    int ICPAfterPointCloudSamplingTest();
    int PointCloudRegistrationTestUsingPCL();
}

using namespace Tester;
int main(int argc, char** argv)
{
    return PointCloudSamplingTest();
    //return ICPAfterPointCloudSamplingTest();
    //return PointCloudRegistrationTestUsingPCL();
}