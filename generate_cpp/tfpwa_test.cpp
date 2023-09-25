#include <stdio.h>  
#include <stdlib.h>   // EXIT_FAILURE
#include <dlfcn.h>    // dlopen, dlerror, dlsym, dlclose
#include<iostream>

#include "tfpwa_model.h" 


int main()
{
    TFPWA_Model a =  TFPWA_Model("model1");
    std::vector<double> p0; 
    p0.push_back(1.999647654683577747e+00);
    p0.push_back(3.673691873692898069e-01),
    p0.push_back(-5.131577891098971778e-01),
    p0.push_back(-3.235870578824232013e-01);
    std::vector<double> p1;
    p1.push_back(1.133887889829692908e+00);
    p1.push_back(-2.083611643784952505e-01);
    p1.push_back(9.667642588050698871e-01);
    p1.push_back(-2.528581805070886923e-01);
    std::vector<double> p2;
    p2.push_back(1.979976191905678418e+00);
    p2.push_back(-1.187931898395349628e-01);
    p2.push_back(-4.149033662170539261e-01);
    p2.push_back(5.064169886198701676e-01);
    std::vector<double> p3;
    p3.push_back(1.658282619339445929e-01);
    p3.push_back(-4.021483302920603187e-02);
    p3.push_back(-3.870310312993303220e-02);
    p3.push_back(7.002824932716572581e-02);
    std::vector<std::vector<double> > p;
   p.push_back(p0);
   p.push_back(p1);
   p.push_back(p2);
   p.push_back(p3) ; //  = {p0, p1, p2, p3};
    std::cout << a.eval(p) << std::endl;
} 


