// CppFlow headers
#include <cppflow/ops.h>
#include <cppflow/model.h>

// C++ headers
#include <iostream>

int main() {

    auto p_0 = cppflow::tensor({2.19771882,  0.73008701, -0.55660393, -0.70110049});
    auto p_1 = cppflow::tensor({0.77582355, -0.19216696,  0.4872549 , -0.2895509});
    auto p_2 = cppflow::tensor({2.12028122, -0.48871821,  0.04448316,  0.8815766});
    auto p_3 = cppflow::tensor({0.18551641, -0.04920183,  0.02486587,  0.1090748});
    cppflow::model model("model2/");

    auto output = model({{"serving_default_p_0:0", p_0},
                        {"serving_default_p_1:0", p_1},
                        {"serving_default_p_2:0", p_2},
                        {"serving_default_p_3:0", p_3}},
                        {"StatefulPartitionedCall:0",});

    auto data = output[0].get_data<double>();

    std::cout << "weight: " << data[0] << std::endl;
    return 0;
}
