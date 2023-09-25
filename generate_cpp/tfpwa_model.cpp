#include <stdio.h>  
#include <stdlib.h>   // EXIT_FAILURE
#include <dlfcn.h>    // dlopen, dlerror, dlsym, dlclose
#include<iostream>

#include "tfpwa_model.h" 


// 定义函数指针类型的别名
typedef double _pt[1][4];
typedef double(* FUNC_AMP)(_pt*);

namespace TFPWA {

class dyn_model {
public:
    dyn_model(std::string path) {
        handle = dlopen(path.c_str(), RTLD_LAZY );
        if( !handle )
        {
            handle = dlopen((path + "/dyn_model.so").c_str(), RTLD_LAZY );
            if( !handle )
            {
                fprintf( stderr, "[%s](%d) dlopen get error: %s\n", __FILE__, __LINE__, dlerror() );
                exit( EXIT_FAILURE );
            }
        }
        amp_func = (FUNC_AMP)dlsym( handle, "amp" );
        // std::cout << (void*)amp_func << std::endl;
        if (!amp_func) {
            std::cout << "not found" << amp_func << std::endl;
        }
    }
    
    ~dyn_model() {
        if (handle) {
            dlclose(handle);
        }
    }
    
    double eval_amplitude(std::vector<std::vector<double> > p) {
        _pt* pv = new _pt[p.size()];
        for (int i=0;i<p.size();i++){
            pv[i][0][0] = p[i][0];
            pv[i][0][1] = p[i][1];
            pv[i][0][2] = p[i][2];
            pv[i][0][3] = p[i][3];
        }
        // std::cout << (void*)amp_func << std::endl;
        double ret = amp_func(pv);
        return ret;
    }
    

    
    void* handle;
    FUNC_AMP amp_func;
    
};

}

TFPWA_Model::TFPWA_Model(std::string name) {
    my_model = new TFPWA::dyn_model(name);
}

TFPWA_Model::~TFPWA_Model() {
    if (my_model) {
        delete my_model;
    }
}


double TFPWA_Model::eval(std::vector<std::vector<double> > p) {
    return my_model->eval_amplitude(p);
}
