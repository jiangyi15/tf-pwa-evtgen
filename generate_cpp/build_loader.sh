
MODEL_DIR=./
g++ ${MODEL_DIR}/tfpwa_model.cpp -fPIC -shared -o ${MODEL_DIR}/libtfpwa_model.so -I ${MODEL_DIR} -ldl
g++ ${MODEL_DIR}/tfpwa_test.cpp -I ${MODEL_DIR} -L ${MODEL_DIR} -ldl -ltfpwa_model 





