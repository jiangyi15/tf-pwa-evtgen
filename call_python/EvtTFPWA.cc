#include "EvtGenModels/EvtTFPWA.hh"
#include "EvtGenBase/EvtParticle.hh"
#include "EvtGenBase/EvtRandom.hh"
#include "EvtGenBase/EvtVector4R.hh"
#include <iostream>
#include <vector>

PyObject *vector2tuple(std::vector<double *> all_p) {
  PyObject *ret = PyTuple_New(all_p.size());
  for (size_t i = 0; i < all_p.size(); i++) {
    double *pi = all_p[i];
    PyObject *tmp = PyTuple_New(4);
    for (int j = 0; j < 4; j++) {
      PyTuple_SetItem(tmp, j, Py_BuildValue("f", pi[j]));
    }
    PyTuple_SetItem(ret, i, tmp);
  }
  return ret;
}

__attribute((constructor)) void _init_python() {
  if (!Py_IsInitialized()) {
    Py_SetPath(
        L"condaenv/lib/python38.zip:"
        L"condaenv/lib/python3.8/lib-dynload:"
        L"condaenv/lib/python3.8/:"
        L"condaenv/lib:"
		L"condaenv/lib/python3.8/site-packages/");
    Py_Initialize();
  }
}

__attribute((deconstructor)) void _deinit_python() { Py_Finalize(); }

EvtTFPWA::~EvtTFPWA() {}

void EvtTFPWA::getName(std::string &model_name) {

  model_name = "TFPWA";
  // return model_name;
}

EvtDecayBase *EvtTFPWA::clone() { return new EvtTFPWA; }

void EvtTFPWA::init() {

  // check that there are 4 arguments: Invariant mass part. Index: i,j, histor.
  // file name, Hid checkNArg(0);

  // bool idSigmap  = getDaugs()[0]==EvtPDL::getId(std::string("Sigma+"))  ||
  // getDaugs()[0]==EvtPDL::getId(std::string("anti-Sigma-")); bool idPi0 =
  // getDaugs()[1]==EvtPDL::getId(std::string("pi0")); if(!(idSigmap && idPi0 )
  // ){std::cout<<"EvtTFPWA: the daughter sequence should be Sigmap
  // Pi0"<<std::endl;abort();}

  // EvtSpinType::spintype parenttype = EvtPDL::getSpinType(getParentId());

  // setup python env
  //
  // invoke python cmd
  PyRun_SimpleString("import sys");
  PyRun_SimpleString("print (\"---import sys---\")");
  PyRun_SimpleString("sys.path.append('./')");
  // invoke specific python script
  pName = PyUnicode_FromFormat("cal_amp");
  pModule = PyImport_Import(pName);
  if (!pModule) {
    std::cout << "Fail to load cal_amp.py" << std::endl;
    PyErr_Print();
    return;
  }

  // get specific function or class in mytest.py
  pDict = PyModule_GetDict(pModule);
  if (!pDict) {
    std::cout << "Fail to load dict" << std::endl;
    return;
  }

  std::cout << "----------------------" << std::endl;

  pFunc = PyDict_GetItemString(pDict, "do_weight");
  if (!pFunc) {
    std::cout << "Fail to load function do_weight" << std::endl;
    return;
  }

  // pInstance = PyInstanceMethod_New(pClass);
  // PyObject* ret = PyObject_CallMethod(pInstance, "prepare", "O", pInstance);
}

void EvtTFPWA::initProbMax() { noProbMax(); }

typedef double Lv[4];

void EvtTFPWA::decay(EvtParticle *p) {

loop:
  p->initializePhaseSpace(getNDaug(), getDaugs());

  EvtParticle *par;
  EvtVector4R p4;
  std::vector<double *> allp;
  Lv *tmp = new Lv[getNDaug()];

  for (int i = 0; i < getNDaug(); i++) {
    par = p->getDaug(i);
    p4 = par->getP4();
    for (int j = 0; j < 4; j++)
      tmp[i][j] = p4.get(j);
    allp.push_back(tmp[i]);
  }

  PyObject *pArgs = PyTuple_New(1);
  PyObject *input_p = vector2tuple(allp);
  //  PyTuple_SetItem(input_p, 3, input_p3);
  PyTuple_SetItem(pArgs, 0, input_p);
  // 调用Python函数

  PyObject *Ret = PyObject_CallObject(pFunc, pArgs);
  double weight = PyFloat_AsDouble(Ret);
  // std::cout << weight << std::endl;
  delete[] tmp;
  if (weight == 0)
    goto loop;
  double rd = EvtRandom::Flat(0.0, 1.0);
  if (rd > weight)
    goto loop; // single out event

  return;
}

void EvtTFPWA::finalize() {
  Py_DECREF(pName);
  Py_DECREF(pModule);
  Py_DECREF(pDict);
  Py_DECREF(pFunc);
}
