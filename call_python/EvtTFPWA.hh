#ifndef EVTTFPWA_HH
#define EVTTFPWA_HH

#include "EvtGenBase/EvtDecayAmp.hh"
#include <python3.8/Python.h>
#include <string>

class Evtparticle;

class EvtTFPWA : public EvtDecayAmp {
public:
  ~EvtTFPWA();
  void getName(std::string &model_name);
  EvtDecayBase *clone();

  void decay(EvtParticle *p);
  void initProbMax();
  void init();
  void finalize();

private:
  PyObject *pName, *pModule, *pDict, *pFunc;
};

#endif
