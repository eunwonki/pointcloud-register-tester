#pragma once

#ifndef EXAMPLE_H
#define EXAMPLE_H

#include "DataType.h"

namespace {
    RegistrationTestData fail1 = RegistrationTestData
    {
        "1234",
        "fail1",
        "model.obj",
        "scene.obj",
        new double[16]
    {
        -0.9943239688873291,
                0.033070243895053864,
                -0.10112399607896805,
                0.34439527988433838,
                -0.042541451752185822,
                0.7476012110710144,
                0.66278398036956787,
                -0.46317532658576965,
                0.097518742084503174,
                0.66332399845123291,
                -0.74195104837417603,
                -0.23278278112411499,
                0,
                0,
                0,
                1
    }
    };

    RegistrationTestData fail2 = RegistrationTestData
    {

    };

    RegistrationTestData fail3 = RegistrationTestData
    {

    };

    RegistrationTestData success2 = RegistrationTestData
    {

    };

    RegistrationTestData success3 = RegistrationTestData
    {

    };
}

#endif