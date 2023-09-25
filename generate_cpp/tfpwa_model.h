#include <string>
#include <vector>

#ifndef TFPWA_MODEL_H_
#define TFPWA_MODEL_H_


namespace TFPWA {
    class dyn_model;
}

class TFPWA_Model {
public:
    TFPWA_Model(std::string path);
    ~TFPWA_Model();
    
    double eval(std::vector<std::vector<double> > all_v);
    TFPWA::dyn_model* my_model;
}; 


#endif
