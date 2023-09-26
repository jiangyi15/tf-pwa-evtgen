#include "EvtGenModels/EvtTFPWA.hh"
// #include "EvtGenBase/EvtVector2tuple.hh"
#include <iostream>
#include "EvtGenBase/EvtParticle.hh"
#include "EvtGenBase/EvtVector4R.hh"
#include "EvtGenBase/EvtRandom.hh"
#include <vector>
#include "EvtGenBase/EvtPDL.hh"
#include "tfpwa_model.h"


EvtTFPWA::~EvtTFPWA() {}

void EvtTFPWA::getName(std::string& model_name){
	model_name="TFPWA";
	// return model_name;
}

EvtDecayBase* EvtTFPWA::clone(){
	return new EvtTFPWA;
}


void EvtTFPWA::init(){
	// check that there are 4 arguments: Invariant mass part. Index: i,j, histor. file name, Hid
	// checkNArg(0);

	// bool idSigmap  = getDaugs()[0]==EvtPDL::getId(std::string("Sigma+"))  || getDaugs()[0]==EvtPDL::getId(std::string("anti-Sigma-"));
	// bool idPi0 = getDaugs()[1]==EvtPDL::getId(std::string("pi0"));
	// if(!(idSigmap && idPi0 ) ){std::cout<<"EvtTFPWA: the daughter sequence should be Sigmap Pi0"<<std::endl;abort();}

	// EvtSpinType::spintype parenttype = EvtPDL::getSpinType(getParentId());

	// TFPWA <*.do lib path> [<max prob>]
	int narg = getNArg();
	if (narg < 1) {
		std::cout << "need model path" << std::endl;
	}
	std::string model_path = getArgStr(0);
	if (narg ==2) {
		max_prob = getArg(1);
	} else {
		max_prob = 1.;
	}

	this->my_model = new TFPWA_Model(model_path);
}

void EvtTFPWA::initProbMax(){
	noProbMax();
}

typedef double Lv[4];

void EvtTFPWA::decay( EvtParticle *p ){

loop:
	p->initializePhaseSpace(getNDaug(),getDaugs());

	EvtParticle *par;
	EvtVector4R p4;

	// add p trans for anti particle
	int charge_id = 1;
	std::string head = "anti";
	std::string s = EvtPDL::name(p->getId());
	if (s.compare(0, head.size(), head) == 0)
		charge_id = -1;

	std::vector<std::vector<double> > inputs;
	for (int i=0;i<getNDaug(); i++) {
		par = p->getDaug(i);
		p4 = par->getP4();
		std::vector<double> p4_v;
		p4_v.push_back(p4.get(0));
		p4_v.push_back( charge_id * p4.get(1));
		p4_v.push_back( charge_id * p4.get(2));
		p4_v.push_back( charge_id * p4.get(3));
		inputs.push_back(p4_v);
	}
                
	double weight = this->my_model->eval(inputs) / max_prob;
	// std::cout << weight << std::endl;
	//	delete[] tmp;
	if(weight == 0) goto loop;
	double rd = EvtRandom::Flat(0.0, 1.0);
	if(rd > weight) goto loop;  //single out event

	return ;
}

void EvtTFPWA::finalize(){
    delete my_model;
}


