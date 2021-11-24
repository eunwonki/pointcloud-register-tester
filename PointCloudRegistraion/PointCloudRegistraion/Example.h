#pragma once

#ifndef EXAMPLE_H
#define EXAMPLE_H

#include "DataType.h"

using namespace std;
using namespace cv;

namespace {
    RegistrationTestData sample = RegistrationTestData
    {
        "20211112114205",
        "sample",
        "data\\model.obj",
        "data\\model.obj",
        new double[16]
    {
        1, 0, 0, 0.02,
        0, 1, 0, -0.01,
        0, 0, 1, -0.03,
        0, 0, 0, 1
    }
    };

    RegistrationTestData fail1 = RegistrationTestData
    {
        "20211112114205",
        "fail1",
        "data\\model.obj",
        "data\\20211112114205\\scene.obj",
        new double[16]
    {
        -0.9943239688873291, 0.033070243895053864, -0.10112399607896805, 0.34439527988433838,
        -0.042541451752185822, 0.7476012110710144, 0.66278398036956787, -0.46317532658576965,
        0.097518742084503174, 0.66332399845123291, -0.74195104837417603, -0.23278278112411499,
        0, 0, 0, 1
    },
        std::vector<cv::Vec3f>
    {
        cv::Vec3f(0.22030885517597198, -0.18102709949016571, -0.21057319641113281)
    },
         std::vector<cv::Vec3f>
        {
            cv::Vec3f(0.113545f, 0.221563f, 0.183073f)
                //Vec3f(0.144233f, 0.194138f, 0.211906f),
                //Vec3f(0.072768f, 0.191650f, 0.210047f),
        },
    };

    RegistrationTestData fail2 = RegistrationTestData
    {
        "20211112114340",
        "fail2",
        "data\\model.obj",
        "data\\20211112114340\\scene.obj",
        new double[16]
    {
        -0.215495094656944275, -0.515701830387115479, 0.829224765300750732, 0.202815130352973938,
        -0.003272652626037598, 0.849552750587463379, 0.527493298053741455, -0.462020576000213623,
        -0.976499497890472412, 0.110958471894264221, -0.184762194752693176, -0.098988197743892670,
        0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 1.000000000000000000
    },
        std::vector<cv::Vec3f>
    {
        cv::Vec3f(0.21589498221874237, -0.17759290337562561, -0.21910548210144043)
        },
        std::vector<cv::Vec3f>
        {
            cv::Vec3f(0.113545f, 0.221563f, 0.183073f)
        },
    };

    RegistrationTestData fail3 = RegistrationTestData
    {
        "20211112114616",
        "fail3",
        "data\\model.obj",
        "data\\20211112114616\\scene.obj",
        new double[16]
        {
          -0.109513953328132629, 0.540356338024139404, -0.834279000759124756, 0.260107129812240601,
          -0.037696596235036850, 0.836465716361999512, 0.546720981597900391, -0.470219850540161133,
           0.993270277976989746, 0.091323055326938629, -0.071235023438930511, -0.327591836452484131,
           0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 1.000000000000000000
        },
        std::vector<cv::Vec3f>
        {
             cv::Vec3f(0.2146613746881485, -0.1890803724527359, -0.20761840045452118)
        },
        std::vector<cv::Vec3f>
        {
            cv::Vec3f(0.113545f, 0.221563f, 0.183073f)
        },
    };

    RegistrationTestData success1 = RegistrationTestData
    {
        "20211112114009",
        "success1",
        "data\\model.obj",
        "data\\20211112114009\\scene.obj",
        new double[16] {
            -0.968687951564788818, -0.222354128956794739, -0.110461480915546417, 0.390509903430938721,
            -0.072253987193107605, -0.173181593418121338, 0.982235789299011230, -0.310296297073364258,
            -0.237534105777740479, 0.959461629390716553, 0.151692986488342285, -0.403522670269012451,
             0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 1.000000000000000000
        },
        std::vector<cv::Vec3f>
        {
            cv::Vec3f(0.21544723212718964, -0.17464002966880798, -0.19083625078201294)
        },
        std::vector<cv::Vec3f>
        {
            cv::Vec3f(0.113545f, 0.221563f, 0.183073f)
        },
    };

    RegistrationTestData success2 = RegistrationTestData
    {
        "20211112114041",
        "success2",
        "data\\model.obj",
        "data\\20211112114041\\scene.obj",
        new double[16] {
            -0.000000043711388287, 0.000000000000000000, 1.000000000000000000, 0.061966955661773682,
             0.000000000000000000, 1.000000000000000000, 0.000000000000000000, -0.394280374050140381,
            -1.000000000000000000, 0.000000000000000000, -0.000000043711388287, -0.117265284061431885,
             0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 1.000000000000000000
        },
        std::vector<cv::Vec3f>
        {
            cv::Vec3f(0.24503993988037109, -0.1727173775434494, -0.23081028461456299)
        },
        std::vector<cv::Vec3f>
        {
            cv::Vec3f(0.113545f, 0.221563f, 0.183073f)
        },
    };

    RegistrationTestData success3 = RegistrationTestData
    {
        "20211112114423",
        "success3",
        "data\\model.obj",
        "data\\20211112114423\\scene.obj",
        new double[16]
        {
           -0.864170193672180176, -0.480944097042083740, -0.147993370890617371, 0.429754197597503662,
           -0.276926219463348389, 0.208984822034835815, 0.937889575958251953, -0.386700749397277832,
           -0.420144110918045044, 0.851479887962341309, -0.313784599304199219, -0.285668551921844482,
           0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 1.000000000000000000
         },
        std::vector<cv::Vec3f>
        {
            cv::Vec3f(0.21018689870834351, -0.1941569596529007, -0.20214903354644775)
        },
        std::vector<cv::Vec3f>
        {
            cv::Vec3f(0.113545f, 0.221563f, 0.183073f)
        },
    };
}

#endif