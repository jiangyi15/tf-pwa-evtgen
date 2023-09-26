#ifndef EVTTFPWA_HH
#define EVTTFPWA_HH

#include "EvtGenBase/EvtDecayAmp.hh"
#include <string>
#include "tfpwa_model.h"


class Evtparticle;

class EvtTFPWA : public EvtDecayAmp {
  public:
    ~EvtTFPWA();
    void getName(std::string& model_name);
    EvtDecayBase* clone();

    void decay( EvtParticle* p );
    void initProbMax();
    void init();
    void finalize();

  private:
    TFPWA_Model* my_model;
    double max_prob;
};

#endif


