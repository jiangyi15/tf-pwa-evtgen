
MODEL_DIR=./model1
g++ ${MODEL_DIR}/temple.cpp -fPIC -shared -o ${MODEL_DIR}/dyn_model.so -I ${MODEL_DIR} -DDEBUG=0


