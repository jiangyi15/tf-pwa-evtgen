#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import \
    convert_variables_to_constants_v2
from tf_pwa.amp import interpolation
from tf_pwa.config_loader import ConfigLoader

from tensorflow.python.framework import tensor_util

import tensorflow.compat.v1 as tf_v1
import subprocess
import os

config = ConfigLoader("config.yml")
config.set_params("a_params.json")
dir_name = "./model1/"

if not os.path.exists(dir_name):
    os.makedirs(dir_name)

n_input_p = len(config.get_dat_order())

phsp_test = config.generate_phsp(10)
p_test = [phsp_test.get_momentum(i) for i in config.get_dat_order()]
# p1, p2, p3, p4 = np.loadtxt("data.dat").reshape((-1,4,4)).transpose((1,0,2))
config.eval_amplitude(*p_test)

pv_sym = [tf.TensorSpec([None, 4], tf.float64) for i in range(n_input_p)]
# p1, p2, p3, p4 = [tf.convert_to_tensor(i) for i in [p1, p2, p3, p4]]

f = tf.function(config.eval_amplitude).get_concrete_function(*pv_sym)
g = convert_variables_to_constants_v2(f)

# print(g(p1, p2, p3, p4))

gd = g.graph.as_graph_def()



# In[2]:


with open(f"{dir_name}/gd.pbtxt", "w") as f:
    print(gd, file=f)


# In[3]:


a = [i.numpy()[:1] for i in p_test]
g(*[tf.convert_to_tensor(i) for i in a])



# In[ ]:





# In[4]:


def get_sort_node(g):
    all_nodes = list(g.node)
    node_order = {}
    used_nodes = []
    for i in all_nodes:
        if len(i.input) == 0: 
            used_nodes.append(i.name)
            node_order[i.name] = 0
    all_nodes = [i for i in all_nodes if i.name not in used_nodes and i.op !="NoOp"]
    while all_nodes:
        used_nodes = []
        for i in all_nodes:
            if all([j in node_order or j.startswith("^") for j in i.input]):
                # print("found", i)
                if i.op == "Unpack":
                    used_nodes.append(i.name)
                    node_order[i.name] = max([node_order[j] for j in i.input if not j.startswith("^")])+1
                    for k in range(i.attr["num"].i):
                        node_order[i.name+":"+str(k)] = max([node_order[j] for j in i.input if not j.startswith("^")])+1
                else:
                    used_nodes.append(i.name)
                    node_order[i.name] = max([node_order[j] for j in i.input if not j.startswith("^")])+1
        if len(used_nodes) == 0:
            raise IndexError("not connect for {}".format([i.name for i in all_nodes if not j.startswith("^")]))
        all_nodes = [i for i in all_nodes if i.name not in used_nodes]
    return node_order

def get_sort_node2(g):
    node_order = get_sort_node(g)
    new_node = [i for i in g.node if i.name in node_order]
    return sorted(new_node, key = lambda x: node_order[x.name])
new_node = get_sort_node2(gd)

used_nodes = []
for i in new_node:
    assert all([j in used_nodes for j in i.input if not j.startswith("^")])
    used_nodes.append(i.name)
    if i.op == "Unpack":
        for k in range(i.attr["num"].i):
            used_nodes.append(i.name+":"+str(k))


# In[5]:


def infer_shape(g, nodes):
    shape_dic = {}
    for i in nodes:
        if i.name == "NoOp":
            continue
        shape_dic[i.name] = g.graph.get_tensor_by_name(i.name+":0").shape
        if i.op == "Unpack":
            for k in range(i.attr["num"].i):
                shape_dic[i.name +":"+str(k)] =  g.graph.get_tensor_by_name(i.name +":"+str(k)).shape
    return shape_dic

def infer_type(g, nodes):
    type_dic = {}
    for i in nodes:
        if i.name == "NoOp":
            continue
        type_dic[i.name] = g.graph.get_tensor_by_name(i.name+":0").dtype
        if i.op == "Unpack":
            for k in range(i.attr["num"].i):
                type_dic[i.name +":"+str(k)] =  g.graph.get_tensor_by_name(i.name +":"+str(k)).dtype
    return type_dic




# g.graph.get_tensor_by_name("unstack_8").shape

shape_dic = infer_shape(g, new_node)
type_dic = infer_type(g, new_node)

#shape_dic[new_node[-5].name]
#for i in new_node:
    #print(i.name , shape_dic[i.name])


# In[6]:


input_position_var = ["p_{}".format(i) for i in range(n_input_p)]

with tf_v1.Session() as sess:
    tf_v1.import_graph_def(gd)
    # print(sess.graph.as_graph_def())i
    val = [sess.graph.get_tensor_by_name("import/"+i.name +":0") for i in new_node if i.name not in input_position_var and ":" not in i.name and i.name != "NoOp"]
    feed_dict = {"import/p_{}:0".format(i): a[i] for i in range(n_input_p)}
    all_val = sess.run(val, feed_dict)


# In[7]:


type_map = {
    tf.float64: "double",
    tf.int32: "long",
    tf.float32: "float",
    tf.complex128: "complex",
    tf.bool: "bool",
}


def ov(name):
    return  name.replace("/", "__").replace(":", "_s_")

def ot(name, n=1):
    return type_map[type_dic[name]] +  "".join(["[{}]".format(k if k else 1) for k in shape_dic[name]])

otr_dic = {}
def otr(name, n=1):
    shape = tuple([i for i in shape_dic[name]])
    key = (type_map[type_dic[name]], shape)
    if key not in otr_dic:
        otr_dic[key] = "".join(["[{}]".format(k if k else n) for k in shape])
    return key[0] + otr_dic[key].replace("[", "_l_").replace("]", "_r_")
    # return type_map[type_dic[name]] + "[]" + "".join(["[{}]".format([(k if k else n) for k in ]))


def print_alloc_array(shape_dic, inputs, n=1):
    ret = ""
    for k,v in shape_dic.items():
        if k in inputs:
            continue
        ret += "{} {};".format(otr(k), ov(k)) +"\n"
    return ret
code1 = print_alloc_array(shape_dic, ["p_0", "p_1", "p_2", "p_3"])
# print(code1)


# In[8]:


def def_call(new_node, type_dic, inputs):
    ret = ""
    for i in new_node:
        if i.name in inputs:
            continue
        params = list(i.input) 
        if i.op != "Unpack":
            ret_params = [ i.name ]
        else:
            ret_params = [ i.name if k==0 else i.name + ":" +str(k) for k in range(i.attr["num"].i) ]
        ret += "{}({});".format(ov(i.name) + "_"+i.op, ",".join(["&"+ov(j) for j in params+ret_params if not j.startswith("^")])) + "\n"
        ret += "if (DEBUG) {{std::cout << \" {}\" <<\" \"<< ".format(i.name)
        for k in range(shape_size(shape_dic[i.name])):
            ret += " <<\" \"<< ".join(["*(({}*) (&{}) + {})".format(type_map[type_dic[j]], ov(j), k) for j in ret_params]) +"<<\" \" << "
        ret += "std::endl;}\n"
    return ret


def shape_size(shape):
    a = 1
    for i in shape:
        if i is not None:
            a = a * i
    return a


code2 = def_call(new_node, type_dic,["p_0", "p_1", "p_2", "p_3"])


# In[9]:


def shape_one(name1, name2):
    if isinstance(name1, str):
        shape1 = shape_dic[name1]
    else:
        shape1 = name1
    if isinstance(name2, str):
        shape2 = shape_dic[name2]
    else:
        shape2 = name2
    shape1 = tuple([i if i else 1 for i in shape1])
    shape2 = tuple([i if i else 1 for i in shape2])
    
    
    def broadcast_index(idx1, idx2):
        if len(idx1) == 0 and len(idx2) == 0:
            yield [], [], []
        elif len(idx1) == 0:
            nb = idx2[0]
            for i, j, p in broadcast_index([], idx2[1:]):
                for k in range(nb):
                    yield [], [k] + j, [k]+p
        elif len(idx2) == 0:
            na = idx1[0]
            for i, j, p in broadcast_index(idx1[1:], []):
                for k in range(na):
                    yield [k] + i, j, [k] + p
        else:
            na = idx1[0]
            nb = idx2[0]
            
            assert max(na, nb) % na == 0
            assert max(na, nb) % nb == 0
            
            for i, j, p in broadcast_index(idx1[1:], idx2[1:]):
                for k in range(max(nb, na)):
                    yield [k % na] + i, [k % nb] + j, [k] + p
    
    for i, j, p in broadcast_index(shape1[::-1], shape2[::-1]):
        yield i[::-1], j[::-1], p[::-1]

def shape_two(name1, name2, name3):
    if isinstance(name1, str):
        shape1 = shape_dic[name1]
    else:
        shape1 = name1
    if isinstance(name2, str):
        shape2 = shape_dic[name2]
    else:
        shape2 = name2
    if isinstance(name3, str):
        shape3 = shape_dic[name3]
    else:
        shape3 = name3
    shape1 = tuple([i if i else 1 for i in shape1])
    shape2 = tuple([i if i else 1 for i in shape2])
    shape3 = tuple([i if i else 1 for i in shape3])
    
    
    def broadcast_index2(idx1, idx2):
        if len(idx1) == 0 and len(idx2) == 0:
            yield [], [], []
        elif len(idx1) == 0:
            nb = idx2[0]
            for i, j, p in broadcast_index2([], idx2[1:]):
                for k in range(nb):
                    yield [], [k] + j, [k]+p
        elif len(idx2) == 0:
            na = idx1[0]
            for i, j, p in broadcast_index2(idx1[1:], []):
                for k in range(na):
                    yield [k] + i, j, [k] + p
        else:
            na = idx1[0]
            nb = idx2[0]
            
            assert max(na, nb) % na == 0
            assert max(na, nb) % nb == 0
            
            for i, j, p in broadcast_index2(idx1[1:], idx2[1:]):
                for k in range(max(nb, na)):
                    yield [k % na] + i, [k % nb] + j, [k] + p
    
    
    
    def broadcast_index(idx1, idx2, idx3):
        if len(idx1) == 0 and len(idx2) == 0 and len(idx3)==0:
            yield [], [], [], []
        elif len(idx1) == 0:
            for i, j, p in broadcast_index2(idx2, idx3):
                yield [], i, j, p
        elif len(idx2) == 0:
            for i, j, p in broadcast_index2(idx1, idx3):
                yield i, [], j, p
        elif len(idx3) == 0:
            for i, j, p in broadcast_index2(idx1, idx2):
                yield i, j, [], p
        else:
            na = idx1[0]
            nb = idx2[0]
            nc = idx3[0]            
            assert max(na, nb, nc) % na == 0
            assert max(na, nb, nc) % nb == 0
            assert max(na, nb, nc) % nc == 0

            for i, j, l, p in broadcast_index(idx1[1:], idx2[1:], idx3[1:]):
                for k in range(max(nb, na, nc)):
                    yield [k % na] + i, [k % nb] + j, [k%nc]+l, [k] + p
    
    for i, j, l, p in broadcast_index(shape1[::-1], shape2[::-1], shape3[::-1]):
        yield i[::-1], j[::-1], l[::-1], p[::-1]

def idx_bd(idx):
    return "".join(["[{}]".format(i) for i in idx])

#for i, j, k in shape_one([2,3], [3,2,1]):
    #print(idx_bd(i), idx_bd(j), idx_bd(k))


# In[10]:



impl_call_dic = {}

def regist_call_impl(name):
    def f(g):
        impl_call_dic[name] = g
    return f

@regist_call_impl("Const")
def impl_const(i, params):
    value = tensor_util.MakeNdarray(i.attr["value"].tensor)
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    def set_const(var, val, shape):
        if len(shape) == 0:
            if "complex" in str(val.dtype):
                return "{} = complex({},{});\n".format(var, np.real(val), np.imag(val))
            if "bool" in str(val.dtype):
                return "{} = {};\n".format(var, "1" if val else "0")
            else:
                return "{} = {};\n".format(var, val)
        ret2 = ""
        for i in range(shape[0]):
            ret2 += set_const(var + "["+str(i)+"]", val[i], shape[1:])
        return ret2
    ret += set_const("v0_{}[0]".format(ov(i.name)), value, shape_dic[i.name])
    ret += "}\n"
    return ret

@regist_call_impl("Identity")
def impl_const(i, params):
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    def set_const(var, val, shape):
        if len(shape) == 0:
            return "{} = {};\n".format(var, val)
        ret2 = ""
        for i in range(shape[0] if shape[0] else 1):
            ret2 += set_const(var + "["+str(i)+"]", val+ "["+str(i)+"]", shape[1:])
        return ret2
    assert len(params)==2, "need to params"
    var = "v1_{}[0]".format(ov(i.name))
    val = "v0_{}[0]".format(ov(params[0]))
    ret += set_const(var, val, shape_dic[i.name])
    ret += "}\n"
    return ret

@regist_call_impl("Cast")
def impl_const(i, params):
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    assert len(params)==2, "need to params"
    var = "v1_{}[0]".format(ov(i.name))
    val = "v0_{}[0]".format(ov(params[0]))
    t = type_map[type_dic[i.name]]
    to = type_map[type_dic[params[0]]]
    
    def set_const(var, val, shape):
        if len(shape) == 0:
            if t == to:
                return "{} = {};\n".format(var,val)
            else:
                return "{} = ({}) {};\n".format(var,t, val)
        ret2 = ""
        for j in range(shape[0] if shape[0] else 1):
            ret2 += set_const(var + "["+str(j)+"]", val+ "["+str(j)+"]", shape[1:])
        return ret2
    ret += set_const(var, val, shape_dic[i.name])
    ret += "}\n"
    return ret

@regist_call_impl("Reshape")
def impl_const(i, params):
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    assert len(params)==3, "need 3 params"
    var = "v2_{}".format(ov(i.name))
    val = "v0_{}".format(ov(params[0]))
    
    # print(shape_dic[i.name], shape_dic[params[0]])
    assert shape_size(shape_dic[params[0]]) == shape_size(shape_dic[params[-1]]), "same shape" 
    t = type_map[type_dic[i.name]]
    for j in range(shape_size(shape_dic[i.name])):
        ret += " *(({} *){} + {}) = *(({}*){} + {});\n".format(t, var, j,t, val, j)
    ret += "}\n"
    return ret

@regist_call_impl("Squeeze")
def impl_const(i, params):
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    assert len(params)==2, "need 3 params"
    var = "v1_{}".format(ov(i.name))
    val = "v0_{}".format(ov(params[0]))
    t = type_map[type_dic[i.name]]
    for j in range(shape_size(shape_dic[i.name])):
        ret += " *(({} *){} + {}) = *(({}*){} + {});\n".format(t, var, j,t, val, j)
    ret += "}\n"
    return ret

@regist_call_impl("ExpandDims")
def impl_const(i, params):
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    assert len(params)==3, "need 3 params"
    var = "v2_{}".format(ov(i.name))
    val = "v0_{}".format(ov(params[0]))
    t = type_map[type_dic[i.name]]
    for j in range(shape_size(shape_dic[i.name])):
        ret += " *(({} *){} + {}) = *(({}*){} + {});\n".format(t, var, j,t, val, j)
    ret += "}\n"
    return ret

@regist_call_impl("Sin")
def impl_const(i, params):
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    assert len(params)==2, "need 2 params"
    var = "v1_{}".format(ov(i.name))
    val = "v0_{}".format(ov(params[0]))
    t = type_map[type_dic[i.name]]
    for j in range(shape_size(shape_dic[i.name])):
        ret += " *(({} *){} + {}) = sin(*(({}*){} + {}));\n".format(t, var, j,t, val, j)
    ret += "}\n"
    return ret

@regist_call_impl("Cos")
def impl_const(i, params):
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    assert len(params)==2, "need 2 params"
    var = "v1_{}".format(ov(i.name))
    val = "v0_{}".format(ov(params[0]))
    t = type_map[type_dic[i.name]]
    for j in range(shape_size(shape_dic[i.name])):
        ret += " *(({} *){} + {}) = cos(*(({}*){} + {}));\n".format(t, var, j,t, val, j)
    ret += "}\n"
    return ret

@regist_call_impl("Acos")
def impl_const(i, params):
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    assert len(params)==2, "need 2 params"
    var = "v1_{}".format(ov(i.name))
    val = "v0_{}".format(ov(params[0]))
    t = type_map[type_dic[i.name]]
    for j in range(shape_size(shape_dic[i.name])):
        ret += " *(({} *){} + {}) = acos(*(({}*){} + {}));\n".format(t, var, j,t, val, j)
    ret += "}\n"
    return ret



@regist_call_impl("Neg")
def impl_const(i, params):
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    assert len(params)==2, "need 2 params"
    var = "v1_{}".format(ov(i.name))
    val = "v0_{}".format(ov(params[0]))
    t = type_map[type_dic[i.name]]
    for j in range(shape_size(shape_dic[i.name])):
        ret += " *(({} *){} + {}) = - (*(({}*){} + {}));\n".format(t, var, j,t, val, j)
    ret += "}\n"
    return ret


@regist_call_impl("Sqrt")
def impl_const(i, params):
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    assert len(params)==2, "need 2 params"
    var = "v1_{}".format(ov(i.name))
    val = "v0_{}".format(ov(params[0]))
    t = type_map[type_dic[i.name]]
    for j in range(shape_size(shape_dic[i.name])):
        ret += " *(({} *){} + {}) = sqrt(*(({}*){} + {}));\n".format(t, var, j,t, val, j)
    ret += "}\n"
    return ret

@regist_call_impl("Abs")
def impl_const(i, params):
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    assert len(params)==2, "need 2 params"
    var = "v1_{}".format(ov(i.name))
    val = "v0_{}".format(ov(params[0]))
    t = type_map[type_dic[i.name]]
    for j in range(shape_size(shape_dic[i.name])):
        if t == "double" or t=="float":
            ret += " *(({} *){} + {}) = fabs(*(({}*){} + {}));\n".format(t, var, j,t, val, j)
        else:
            ret += " *(({} *){} + {}) = abs(*(({}*){} + {}));\n".format(t, var, j,t, val, j)
    ret += "}\n"
    return ret


@regist_call_impl("Real")
def impl_const(i, params):
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    assert len(params)==2, "need 2 params"
    var = "v1_{}".format(ov(i.name))
    val = "v0_{}".format(ov(params[0]))
    t = type_map[type_dic[i.name]]
    ti = type_map[type_dic[params[0]]]
    for j in range(shape_size(shape_dic[i.name])):
        ret += " *(({} *){} + {}) = std::real(*(({}*){} + {}));\n".format(t, var, j,ti, val, j)
    ret += "}\n"
    return ret


@regist_call_impl("Imag")
def impl_const(i, params):
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    assert len(params)==2, "need 2 params"
    var = "v1_{}".format(ov(i.name))
    val = "v0_{}".format(ov(params[0]))
    t = type_map[type_dic[i.name]]
    ti = type_map[type_dic[params[0]]]
    for j in range(shape_size(shape_dic[i.name])):
        ret += " *(({} *){} + {}) = std::imag(*(({}*){} + {}));\n".format(t, var, j,ti, val, j)
    ret += "}\n"
    return ret



@regist_call_impl("Conj")
def impl_const(i, params):
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    assert len(params)==2, "need 2 params"
    var = "v1_{}".format(ov(i.name))
    val = "v0_{}".format(ov(params[0]))
    t = type_map[type_dic[i.name]]
    for j in range(shape_size(shape_dic[i.name])):
        ret += " *(({} *){} + {}) = conj(*(({}*){} + {}));\n".format(t, var, j,t, val, j)
    ret += "}\n"
    return ret

@regist_call_impl("Floor")
def impl_const(i, params):
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    assert len(params)==2, "need 2 params"
    var = "v1_{}".format(ov(i.name))
    val = "v0_{}".format(ov(params[0]))
    t = type_map[type_dic[i.name]]
    for j in range(shape_size(shape_dic[i.name])):
        ret += " *(({} *){} + {}) = floor(*(({}*){} + {}));\n".format(t, var, j,t, val, j)
    ret += "}\n"
    return ret



@regist_call_impl("Exp")
def impl_const(i, params):
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    assert len(params)==2, "need 2 params"
    var = "v1_{}".format(ov(i.name))
    val = "v0_{}".format(ov(params[0]))
    t = type_map[type_dic[i.name]]
    for j in range(shape_size(shape_dic[i.name])):
        ret += " *(({} *){} + {}) = exp(*(({}*){} + {}));\n".format(t, var, j,t, val, j)
    ret += "}\n"
    return ret

@regist_call_impl("Log")
def impl_const(i, params):
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    assert len(params)==2, "need 2 params"
    var = "v1_{}".format(ov(i.name))
    val = "v0_{}".format(ov(params[0]))
    t = type_map[type_dic[i.name]]
    for j in range(shape_size(shape_dic[i.name])):
        ret += " *(({} *){} + {}) = log(*(({}*){} + {}));\n".format(t, var, j,t, val, j)
    ret += "}\n"
    return ret

@regist_call_impl("Acosh")
def impl_const(i, params):
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    assert len(params)==2, "need 2 params"
    var = "v1_{}".format(ov(i.name))
    val = "v0_{}".format(ov(params[0]))
    t = type_map[type_dic[i.name]]
    for j in range(shape_size(shape_dic[i.name])):
        ret += " *(({} *){} + {}) = acosh(*(({}*){} + {}));\n".format(t, var, j,t, val, j)
    ret += "}\n"
    return ret

@regist_call_impl("Angle")
def impl_const(i, params):
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    assert len(params)==2, "need 2 params"
    var = "v1_{}".format(ov(i.name))
    val = "v0_{}".format(ov(params[0]))
    t = type_map[type_dic[i.name]]
    t1 = type_map[type_dic[params[0]]]
    for j in range(shape_size(shape_dic[i.name])):
        ret += " *(({} *){} + {}) = std::arg(*(({}*){} + {}));\n".format(t, var, j,t1, val, j)
    ret += "}\n"
    return ret




@regist_call_impl("ZerosLike")
def impl_const(i, params):
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    assert len(params)==2, "need 2 params"
    var = "v1_{}".format(ov(i.name))
    val = "v0_{}".format(ov(params[0]))
    t = type_map[type_dic[i.name]]
    for j in range(shape_size(shape_dic[i.name])):
        ret += " *(({} *){} + {}) = 0;\n".format(t, var, j)
    ret += "}\n"
    return ret


@regist_call_impl("StopGradient")
def impl_const(i, params):
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    assert len(params)==2, "need 2 params"
    var = "v1_{}".format(ov(i.name))
    val = "v0_{}".format(ov(params[0]))
    t = type_map[type_dic[i.name]]
    for j in range(shape_size(shape_dic[i.name])):
        ret += " *(({} *){} + {}) = (*(({}*){} + {}));\n".format(t, var, j,t, val, j)
    ret += "}\n"
    return ret



@regist_call_impl("Range")
def impl_const(i, params):
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    assert len(params)==4, "need 4 params"
    var = "v3_{}".format(ov(i.name))
    start = "v0_{}".format(ov(params[0]))
    end = "v1_{}".format(ov(params[1]))
    step = "v2_{}".format(ov(params[2]))
    t = type_map[type_dic[i.name]]
    ret += f"""
    for (int i= 0;i< ((*{end})-(*{start}) + *({step})-1)/(*{step}); i++ ) {{
        *(({t} *) {var} + i) = (*{start}) + (*{step}) * i;
    }}

    """
    ret += "}\n"
    return ret


@regist_call_impl("Mul")
def impl_const(i, params):
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    assert len(params)==3, "need 2 params"
    var = "v2_{}".format(ov(i.name))
    val0 = "v0_{}".format(ov(params[0]))
    val1 = "v1_{}".format(ov(params[1]))
    shape0 = shape_dic[params[0]]
    shape1 = shape_dic[params[1]]
    t = type_map[type_dic[i.name]]
    for m, n, p in shape_one(shape0, shape1):
        ret += " {}[0]{} = {}[0]{} * {}[0]{};\n".format(var, idx_bd(p), val0, idx_bd(m), val1, idx_bd(n))
    ret += "}\n"
    return ret


@regist_call_impl("Add")
def impl_const(i, params):
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    assert len(params)==3, "need 2 params"
    var = "v2_{}".format(ov(i.name))
    val0 = "v0_{}".format(ov(params[0]))
    val1 = "v1_{}".format(ov(params[1]))
    shape0 = shape_dic[params[0]]
    shape1 = shape_dic[params[1]]
    t = type_map[type_dic[i.name]]
    for m, n, p in shape_one(shape0, shape1):
        ret += " {}[0]{} = {}[0]{} + {}[0]{};\n".format(var, idx_bd(p), val0, idx_bd(m), val1, idx_bd(n))
    ret += "}\n"
    return ret


@regist_call_impl("AddV2")
def impl_const(i, params):
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    assert len(params)==3, "need 2 params"
    var = "v2_{}".format(ov(i.name))
    val0 = "v0_{}".format(ov(params[0]))
    val1 = "v1_{}".format(ov(params[1]))
    shape0 = shape_dic[params[0]]
    shape1 = shape_dic[params[1]]
    t = type_map[type_dic[i.name]]
    for m, n, p in shape_one(shape0, shape1):
        ret += " {}[0]{} = {}[0]{} + {}[0]{};\n".format(var, idx_bd(p), val0, idx_bd(m), val1, idx_bd(n))
    ret += "}\n"
    return ret


@regist_call_impl("Sub")
def impl_const(i, params):
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    assert len(params)==3, "need 2 params"
    var = "v2_{}".format(ov(i.name))
    val0 = "v0_{}".format(ov(params[0]))
    val1 = "v1_{}".format(ov(params[1]))
    shape0 = shape_dic[params[0]]
    shape1 = shape_dic[params[1]]
    t = type_map[type_dic[i.name]]
    for m, n, p in shape_one(shape0, shape1):
        ret += " {}[0]{} = {}[0]{} - {}[0]{};\n".format(var, idx_bd(p), val0, idx_bd(m), val1, idx_bd(n))
    ret += "}\n"
    return ret

@regist_call_impl("RealDiv")
def impl_const(i, params):
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    assert len(params)==3, "need 3 params"
    var = "v2_{}".format(ov(i.name))
    val0 = "v0_{}".format(ov(params[0]))
    val1 = "v1_{}".format(ov(params[1]))
    shape0 = shape_dic[params[0]]
    shape1 = shape_dic[params[1]]
    t = type_map[type_dic[i.name]]
    for m, n, p in shape_one(shape0, shape1):
        ret += " {}[0]{} = {}[0]{} / {}[0]{};\n".format(var, idx_bd(p), val0, idx_bd(m), val1, idx_bd(n))
    ret += "}\n"
    return ret


@regist_call_impl("LogicalAnd")
def impl_const(i, params):
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    assert len(params)==3, "need 3 params"
    var = "v2_{}".format(ov(i.name))
    val0 = "v0_{}".format(ov(params[0]))
    val1 = "v1_{}".format(ov(params[1]))
    shape0 = shape_dic[params[0]]
    shape1 = shape_dic[params[1]]
    t = type_map[type_dic[i.name]]
    for m, n, p in shape_one(shape0, shape1):
        ret += " {}[0]{} = {}[0]{} && {}[0]{};\n".format(var, idx_bd(p), val0, idx_bd(m), val1, idx_bd(n))
    ret += "}\n"
    return ret

@regist_call_impl("LogicalOr")
def impl_const(i, params):
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    assert len(params)==3, "need 3 params"
    var = "v2_{}".format(ov(i.name))
    val0 = "v0_{}".format(ov(params[0]))
    val1 = "v1_{}".format(ov(params[1]))
    shape0 = shape_dic[params[0]]
    shape1 = shape_dic[params[1]]
    t = type_map[type_dic[i.name]]
    for m, n, p in shape_one(shape0, shape1):
        ret += " {}[0]{} = {}[0]{} || {}[0]{};\n".format(var, idx_bd(p), val0, idx_bd(m), val1, idx_bd(n))
    ret += "}\n"
    return ret



@regist_call_impl("Complex")
def impl_const(i, params):
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    assert len(params)==3, "need 3 params"
    var = "v2_{}".format(ov(i.name))
    val0 = "v0_{}".format(ov(params[0]))
    val1 = "v1_{}".format(ov(params[1]))
    shape0 = shape_dic[params[0]]
    shape1 = shape_dic[params[1]]
    t = type_map[type_dic[i.name]]
    for m, n, p in shape_one(shape0, shape1):
        ret += " {}[0]{} = complex({}[0]{} , {}[0]{});\n".format(var, idx_bd(p), val0, idx_bd(m), val1, idx_bd(n))
    ret += "}\n"
    return ret



@regist_call_impl("Pow")
def impl_const(i, params):
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    assert len(params)==3, "need 3 params"
    var = "v2_{}".format(ov(i.name))
    val0 = "v0_{}".format(ov(params[0]))
    val1 = "v1_{}".format(ov(params[1]))
    shape0 = shape_dic[params[0]]
    shape1 = shape_dic[params[1]]
    t = type_map[type_dic[i.name]]
    for m, n, p in shape_one(shape0, shape1):
        ret += " {}[0]{} = pow({}[0]{} , {}[0]{});\n".format(var, idx_bd(p), val0, idx_bd(m), val1, idx_bd(n))
    ret += "}\n"
    return ret


@regist_call_impl("Atan2")
def impl_const(i, params):
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    assert len(params)==3, "need 3 params"
    var = "v2_{}".format(ov(i.name))
    val0 = "v0_{}".format(ov(params[0]))
    val1 = "v1_{}".format(ov(params[1]))
    shape0 = shape_dic[params[0]]
    shape1 = shape_dic[params[1]]
    t = type_map[type_dic[i.name]]
    for m, n, p in shape_one(shape0, shape1):
        ret += " {}[0]{} = atan2({}[0]{} , {}[0]{});\n".format(var, idx_bd(p), val0, idx_bd(m), val1, idx_bd(n))
    ret += "}\n"
    return ret

@regist_call_impl("FloorMod")
def impl_const(i, params):
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    assert len(params)==3, "need 3 params"
    var = "v2_{}".format(ov(i.name))
    val0 = "v0_{}".format(ov(params[0]))
    val1 = "v1_{}".format(ov(params[1]))
    shape0 = shape_dic[params[0]]
    shape1 = shape_dic[params[1]]
    t = type_map[type_dic[i.name]]
    for m, n, p in shape_one(shape0, shape1):
        if t == "double":
            ret += " {}[0]{} = {}[0]{} -floor( {}[0]{} / {}[0]{}) * {}[0]{};\n".format(var, idx_bd(p), val0, idx_bd(m), val0, idx_bd(m), val1, idx_bd(n), val1, idx_bd(n))
        else:
            ret += " {}[0]{} = {}[0]{}  % {}[0]{};\n".format(var, idx_bd(p), val0, idx_bd(m), val1, idx_bd(n))
    ret += "}\n"
    return ret


@regist_call_impl("Greater")
def impl_const(i, params):
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    assert len(params)==3, "need 3 params"
    var = "v2_{}".format(ov(i.name))
    val0 = "v0_{}".format(ov(params[0]))
    val1 = "v1_{}".format(ov(params[1]))
    shape0 = shape_dic[params[0]]
    shape1 = shape_dic[params[1]]
    t = type_map[type_dic[i.name]]
    for m, n, p in shape_one(shape0, shape1):
        ret += " {}[0]{} = {}[0]{}  > {}[0]{};\n".format(var, idx_bd(p), val0, idx_bd(m), val1, idx_bd(n))
    ret += "}\n"
    return ret



@regist_call_impl("Less")
def impl_const(i, params):
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    assert len(params)==3, "need 3 params"
    var = "v2_{}".format(ov(i.name))
    val0 = "v0_{}".format(ov(params[0]))
    val1 = "v1_{}".format(ov(params[1]))
    shape0 = shape_dic[params[0]]
    shape1 = shape_dic[params[1]]
    t = type_map[type_dic[i.name]]
    for m, n, p in shape_one(shape0, shape1):
        ret += " {}[0]{} = {}[0]{}  < {}[0]{};\n".format(var, idx_bd(p), val0, idx_bd(m), val1, idx_bd(n))
    ret += "}\n"
    return ret

@regist_call_impl("LessEqual")
def impl_const(i, params):
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    assert len(params)==3, "need 3 params"
    var = "v2_{}".format(ov(i.name))
    val0 = "v0_{}".format(ov(params[0]))
    val1 = "v1_{}".format(ov(params[1]))
    shape0 = shape_dic[params[0]]
    shape1 = shape_dic[params[1]]
    t = type_map[type_dic[i.name]]
    for m, n, p in shape_one(shape0, shape1): 
         ret += " {}[0]{} = {}[0]{}  <= {}[0]{};\n".format(var, idx_bd(p), val0, idx_bd(m), val1, idx_bd(n))
    ret += "}\n"
    return ret


@regist_call_impl("Minimum")
def impl_const(i, params):
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    assert len(params)==3, "need 3 params"
    var = "v2_{}".format(ov(i.name))
    val0 = "v0_{}".format(ov(params[0]))
    val1 = "v1_{}".format(ov(params[1]))
    shape0 = shape_dic[params[0]]
    shape1 = shape_dic[params[1]]
    t = type_map[type_dic[i.name]]
    for m, n, p in shape_one(shape0, shape1):
        ret += " if ({}[0]{}  < {}[0]{}) {{\n".format(val0, idx_bd(m), val1, idx_bd(n))
        ret += " {}[0]{} =  {}[0]{};\n".format(var, idx_bd(p), val0, idx_bd(m))
        ret += " } else {"
        ret += " {}[0]{} = {}[0]{}; }};\n".format(var, idx_bd(p), val1, idx_bd(n))
        
    ret += "}\n"
    return ret


@regist_call_impl("Maximum")
def impl_const(i, params):
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    assert len(params)==3, "need 3 params"
    var = "v2_{}".format(ov(i.name))
    val0 = "v0_{}".format(ov(params[0]))
    val1 = "v1_{}".format(ov(params[1]))
    shape0 = shape_dic[params[0]]
    shape1 = shape_dic[params[1]]
    t = type_map[type_dic[i.name]]
    for m, n, p in shape_one(shape0, shape1):
        ret += " if ({}[0]{}  > {}[0]{}) {{\n".format(val0, idx_bd(m), val1, idx_bd(n))
        ret += " {}[0]{} =  {}[0]{};\n".format(var, idx_bd(p), val0, idx_bd(m))
        ret += " } else {"
        ret += " {}[0]{} = {}[0]{}; }};\n".format(var, idx_bd(p), val1, idx_bd(n))
        
    ret += "}\n"
    return ret


                    


@regist_call_impl("Cross")
def impl_const(i, params):
    # print(i)
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    assert len(params)==3, "need 3 params"
    var = "v2_{}".format(ov(i.name))
    val0 = "v0_{}".format(ov(params[0]))
    val1 = "v1_{}".format(ov(params[1]))
    shape0 = shape_dic[params[0]]
    shape1 = shape_dic[params[1]]
    t = type_map[type_dic[i.name]]
    assert shape0[-1] == shape1[-1]
    assert shape0[-1] == 3
    for m, n, p in shape_one(shape0[:-1], shape1[:-1]):
        ret += " {}[0]{}[0] = {}[0]{}[1]*{}[0]{}[2]  - {}[0]{}[2] * {}[0]{}[1] ;\n".format(var, idx_bd(p), val0, idx_bd(m), val1, idx_bd(n), val0, idx_bd(m), val1, idx_bd(n))
        ret += " {}[0]{}[1] = {}[0]{}[2]* {}[0]{}[0] - {}[0]{}[0] * {}[0]{}[2];\n".format(var, idx_bd(p), val0, idx_bd(m), val1, idx_bd(n), val0, idx_bd(m), val1, idx_bd(n))
        ret += " {}[0]{}[2] = {}[0]{}[0]* {}[0]{}[1]  - {}[0]{}[1] * {}[0]{}[0];\n".format(var, idx_bd(p), val0, idx_bd(m), val1, idx_bd(n), val0, idx_bd(m), val1, idx_bd(n))
    ret += "}\n"
    return ret


@regist_call_impl("StridedSlice")
def impl_const(i, params):
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    assert len(params)==5, "need 3 params"
    var = "v4_{}".format(ov(i.name))
    val0 = "v0_{}".format(ov(params[0]))
    start = "v1_{}".format(ov(params[1]))
    end = "v2_{}".format(ov(params[2]))
    step = "v3_{}".format(ov(params[3]))
    shape0 = shape_dic[params[0]]
    shape1 = shape_dic[params[1]]
    # print(shape0, shape1)
    assert len(shape0) >= shape1[0]
    t = type_map[type_dic[i.name]]
    n = shape1[0]
    ret += "if (DEBUG) {"
    for j in range(n):
        ret += "std::cout << \"idx: \" << {}[0][{}] << \" \"  << {}[0][{}] << \" \"  << {}[0][{}] << \" \"  <<std::endl;".format(start, j, end, j, step, j) 
    ret += "}"
    idx_val = ""
    ret += "int idx = 0;"
    ret += "int size1 = {};".format(shape_size(shape0)//shape_size(shape0[:shape1[0]]))
    for k in range(n):
        start_val = "{}[0][{}]".format(start, k)
        end_val = "{}[0][{}]".format(end, k)
        step_val = "{}[0][{}]".format(step, k)
        nk = shape0[k] if shape0[k] else 1
        ret += f"int n_iter_{k} = (({end_val} - {start_val})/{step_val})%({nk});"
        ret += f"if (n_iter_{k} == 0) n_iter_{k}={shape0[k] if shape0[k] else 1};"
        ret += "  " * k +  f" for (int idx_{k}= 0 ; idx_{k} < n_iter_{k}; idx_{k} ++ ) {{\n"
        idx_val += "[({}+idx_{}*{} + {})%({})]".format(start_val, k, step_val, nk, nk)
    for k in range(n, len(shape0)):
        ret += "  " * k +  f" for (int idx_{k}= 0 ; idx_{k} < {shape0[k]}; idx_{k} ++ ) {{\n"
        idx_val += "[idx_{}]".format(k)    
    ret += "  "*n + "*(({}*){} + idx) = {}[0]{};\n".format(t, var, val0,idx_val)
    ret += "  "*n + "idx ++;\n"
    ret += "}\n" * len(shape0)
    ret += "}\n"
    return ret

@regist_call_impl("SelectV2")
def impl_const(i, params):
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    assert len(params)==4, "need 3 params"
    var = "v3_{}".format(ov(i.name))
    cond1 = "v0_{}".format(ov(params[0]))
    tv = "v1_{}".format(ov(params[1]))
    fv = "v2_{}".format(ov(params[2]))
    shape0 = shape_dic[params[0]]
    shape1 = shape_dic[params[1]]
    shape2 = shape_dic[params[2]]
    t = type_map[type_dic[i.name]]
    tb = type_map[type_dic[params[0]]]
    for j,k, l, p in shape_two(shape0, shape1, shape2):
        ret += "if ({}[0]{}) {{".format(cond1, idx_bd(j) )
        ret += " {}[0]{} = {}[0]{};\n".format(var, idx_bd(p), tv, idx_bd(k))
        ret += "} else { "
        ret += " {}[0]{} = {}[0]{};\n".format(var, idx_bd(p), fv, idx_bd(l))
        ret += "}"
    ret += "}\n"
    return ret

@regist_call_impl("Shape")
def impl_const(i, params):
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    assert len(params)==2, "need 1 params"
    var = "v1_{}".format(ov(i.name))
    cond1 = "v0_{}".format(ov(params[0]))
    shape0 = shape_dic[params[0]]
    shape1 = shape_dic[params[1]]
    t = type_map[type_dic[i.name]]
    for j in range(shape1[0]):
        ret += " *(({} *){} + {}) = {};\n".format(t, var, j, shape0[j] if shape0[j] else 1)
    ret += "}\n"
    return ret

@regist_call_impl("Pack")
def impl_const(i, params):
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    # assert len(params)==2, "need 1 params"
    n_input = len(params) - 1
    var = "v{}_{}".format(n_input, ov(i.name))
    input_var = ["v{}_{}".format(it, ov(j)) for it, j in enumerate(params[:-1])]
    axis = i.attr["axis"].i
    shape0 = shape_dic[params[0]]
    shape1 = shape_dic[i.name]
    axis = (axis +len(shape1)) % len(shape1)
    new_shape = shape0[:axis] + [1] + shape0[axis:]
    t = type_map[type_dic[i.name]]
    for m, n, p in shape_one(shape1, new_shape):
        idx = m[axis]
        ret += " {}[0]{} = {}[0]{};\n".format(var, idx_bd(m), input_var[idx], idx_bd(n[:axis] + n[axis+1:]))
    ret += "}\n"
    return ret


@regist_call_impl("Unpack")
def impl_const(i, params):
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    # assert len(params)==2, "need 1 params"
    n_output = len(params) - 1
    var = "v0_{}".format(ov(params[0]))
    output_var = ["v{}_{}".format(it+1, ov(j)) for it, j in enumerate(params[1:])]
    axis = i.attr["axis"].i
    shape0 = shape_dic[params[0]]
    shape1 = shape_dic[i.name]
    axis = (axis +len(shape0)) % len(shape0)
    # print(axis)
    new_shape = shape1[:axis] + [1] + shape1[axis:]
    t = type_map[type_dic[i.name]]
    for m, n, p in shape_one(shape0, new_shape):
        idx = m[axis]
        ret += " {}[0]{} = {}[0]{};\n".format(output_var[idx], idx_bd(n[:axis] + n[axis+1:]), var, idx_bd(m))
    ret += "}\n"
    return ret


@regist_call_impl("Fill")
def impl_const(i, params):
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    assert len(params)==3, "need 1 params"
    var = "v2_{}".format(ov(i.name))
    shape = "v0_{}".format(ov(params[0]))
    value = "v1_{}".format(ov(params[1]))
    shape0 = shape_dic[params[0]]
    t = type_map[type_dic[i.name]]
    ts = type_map[type_dic[params[0]]]
    tv = type_map[type_dic[params[1]]]
    ret += "int idx=0;"
    n = len(shape0)
    for k in range(n):
        ret += "for (int idx_{}=0; idx_{}< (*(({}*){} + {})) ; idx_{} ++) {{\n".format(k, k, ts, shape, k, k)
    ret += " *(({}*){} + idx) = *(({} *) {});\n".format(t, var, tv, value)
    ret += "idx++;"
    ret += "}" * n
    ret += "}\n"
    return ret




@regist_call_impl("ConcatV2")
def impl_const(i, params):
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    n_input = len(params) - 2
    ret_v = "v{}_{}".format(n_input+1, ov(i.name))
    axis = "v{}_{}".format(n_input, ov(params[-2]))
    inputs = ["v{}_{}".format(i, ov(params[i])) for i in range(n_input)]
    shape0 = [shape_dic[params[i]] for i in range(n_input)]
    shape_all = shape_dic[params[-1]]
    for j, k in enumerate(shape_all):
        if k != shape0[0][j]:
            axis_value = j
            break
    else:
        raise ValueError(" not found")
    
    t = type_map[type_dic[i.name]]
    tv = type_map[type_dic[params[0]]]
    ts = type_map[type_dic[params[-2]]]
    ret += "int axis = ((*(({}*) {}))+{}) % ({});".format(ts, axis, len(shape0), len(shape0))
    size1 = 1
    size2 = 1
    size_all  = 1
    for j, k in enumerate(shape_all):
        if axis_value < j:
            size1 *= k if k else 1
        size_all *= k if k else 1
    ret += "int idx = 0;"
    ret += "for (int i=0;i<{};i++) {{".format((size_all//size1)//shape_all[axis_value])
    for j, name in enumerate(inputs):
        n_j = size1 * shape0[j][axis_value]
        ret += "for (int j=0;j<{}; j++){{\n".format(n_j)
        ret += "*(({}*){} + idx) = *(({}*){} + i * {} + j );\n idx++;\n".format(t, ret_v, tv, name, n_j)
        ret += "}"
    ret += "}"
    ret += "}\n"
    return ret

@regist_call_impl("MatMul")
def impl_const(i, params):
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    assert len(params) == 3, ""
    n_input = 2
    ret_v = "v{}_{}".format(n_input, ov(i.name))
    inputs = ["v{}_{}".format(i, ov(params[i])) for i in range(n_input)]
    shape0 = [shape_dic[params[i]] for i in range(n_input)]
    shape_all = shape_dic[params[-1]]
    assert not i.attr["transpose_a"].b 
    assert not i.attr["transpose_b"].b
    assert len(shape_all) == 2
    sum_size = shape0[0][-1]
    a = shape0[0][0] if shape0[0][0] else 1
    c = shape0[1][-1]
    
    for j in range(a):
        for k in range(c):
            ret += " {}[0][{}][{}] = 0;\n".format(ret_v, j, k)
            for p in range(sum_size):
                ret += " {}[0][{}][{}] += {}[0][{}][{}] * {}[0][{}][{}];\n".format(ret_v, j, k, inputs[0], j, p, inputs[1], p, k)
    ret += "}\n"
    return ret


@regist_call_impl("GatherV2")
def impl_const(i, params):
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    assert len(params) == 4, ""
    n_input = 3
    ret_v = "v{}_{}".format(n_input, ov(i.name))
    inputs = ["v{}_{}".format(i, ov(params[i])) for i in range(n_input)]
    shape0 = [shape_dic[params[i]] for i in range(n_input)]
    type0 = [type_map[type_dic[params[i]]] for i in range(n_input)]
    type_r = type_map[type_dic[i.name]]

    
    shape_all = shape_dic[params[-1]]
    assert len(shape0[1]) == 1, "not list index"
    assert i.attr["batch_dims"].i == 0, "only the first dims"

    input_size = shape_size(shape0[0])
    output_size = shape_size(shape_all)
    ret += "int size1 = 1; int size2 = 1;"
    ret += "int axis =( *(({} *) {}) + {}) % ({});\n".format(type0[2], inputs[2], len(shape0[0]) ,  len(shape0[0]))
    for j in range(len(shape0[0])):
        ret += "if (axis < {}) {{".format(j)
        ret += "size1 *= {}; }}".format(shape0[0][j] if shape0[0][j]  else 1)
        ret += "if (axis == {}) {{".format(j)
        ret += "size2 = {}; }}".format(shape0[0][j] if shape0[0][j]  else 1)
    ret += "int size3 = {}/size1/size2;".format(input_size)
    ret += "int size4 = {}/size1/size3;".format(output_size)
    
    ret += "for (int i=0;i<size3;i++) {\n"
    ret += "for (int j=0;j< size4; j++){\n"
    ret += " int idx = *(({}*) {} + j);\n".format(type0[1], inputs[1])
    ret += "for (int k=0;k<size1; k++) {\n"
    ret += "*(({} *){} + i *size1 * size4 + j * size1 + k ) =".format(type_r, ret_v)
    ret +=  " *(({}*) {} + i * size1 * size2 + idx* size1 + k);\n".format(type0[0], inputs[0])
    ret += "}}}"
    ret += "}\n"
    return ret


@regist_call_impl("Pad")
def impl_const(i, params):
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    assert len(params) == 3, ""
    n_input = 2
    ret_v = "v{}_{}".format(n_input, ov(i.name))
    inputs = ["v{}_{}".format(i, ov(params[i])) for i in range(n_input)]
    shape0 = [shape_dic[params[i]] for i in range(n_input)]
    type0 = [type_map[type_dic[params[i]]] for i in range(n_input)]
    type_r = type_map[type_dic[i.name]]
    shape_all = shape_dic[params[-1]]
    input_size = shape_size(shape0[0])
    output_size = shape_size(shape_all)
    
    ndim = len(shape_all)
    for j in range(ndim):
        ret += "for (int idx_{}=0; idx_{} < {}; idx_{}++) {{".format(j,j,shape_all[j] if shape_all[j] else 1, j)
        ret += "int pad_l_{} = *(({}*) {} + 2* {});".format(j, type0[1], inputs[1], j)
        ret += "int pad_r_{} = *(({}*) {} + 2* {} + 1);".format(j, type0[1], inputs[1], j)
        ret += "if (idx_{} < pad_l_{}) continue;".format(j, j) 
        ret += "if (idx_{} >= pad_l_{} + {}) continue;".format(j, j, shape0[0][j] if shape0[0][j] else 1) 
    ret += "{}[0]{} = {}[0]{};".format(ret_v, "".join(["[idx_{}]".format(k) for k in range(ndim)]), inputs[0], "".join(["[idx_{}-pad_l_{}]".format(k,k) for k in range(ndim)]), )
    ret += "}" * ndim
    ret += "}\n"
    return ret



def impl_sum_1(i, params):
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    assert len(params)==3, "need 2 params"
    var = "v2_{}".format(ov(i.name))
    value = "v0_{}".format(ov(params[0]))
    dims = "v1_{}".format(ov(params[1]))
    shape0 = shape_dic[params[0]]
    shape1 = shape_dic[params[1]]
    t = type_map[type_dic[i.name]]
    tv = type_map[type_dic[params[0]]]
    ts = type_map[type_dic[params[1]]]
    ret += "long axis = *(({} *){});".format(ts, dims)
    ret += "axis = (axis + {})% {};\n".format(len(shape0), len(shape0))
    ret += "long size1 = 1, size2=1;"
    for i, m in enumerate(shape0):
        ret += "if (axis < {}) {{ size1 *= {}; }}".format(i, m if m else 1)
        ret += "if (axis == {}) {{ size2 = {}; }}".format(i, m if m else 1)
    ret += "long size3 = {}/size1 /size2;".format(shape_size(shape0))
    ret += "for (int i=0;i<size3;i++) {\n"
    ret += "for (int k=0; k< size1; k++) {\n"
    ret += "*(({} *) {} + i *size1 + k) = 0;".format(t, var)
    ret += "for (int j=0;j< size2; j++) {\n"
    ret += "*(({} *) {} +  i *size1 + k) += *(({} *) {} +  i *size1*size2 +j *size1 + k);".format(t, var, tv, value)
    ret += "}}}"
    ret += "}\n"
    return ret

def impl_sum_0(i, params):
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    assert len(params)==3, "need 2 params"
    var = "v2_{}".format(ov(i.name))
    value = "v0_{}".format(ov(params[0]))
    dims = "v1_{}".format(ov(params[1]))
    shape0 = shape_dic[params[0]]
    shape1 = shape_dic[params[1]]
    t = type_map[type_dic[i.name]]
    tv = type_map[type_dic[params[0]]]
    ts = type_map[type_dic[params[1]]]
    ret += "for (int i=0;i< {};i++) {{\n".format(shape_size(shape0))
    ret += "*(({} *) {} +  i ) = *(({} *) {} +  i );".format(t, var, tv, value)
    ret += "}"
    ret += "}\n"
    return ret
    

@regist_call_impl("Sum")
def impl_const(i, params):
    ret = "void inline {}({}) {{\n".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)]))
    assert len(params)==3, "need 2 params"
    var = "v2_{}".format(ov(i.name))
    value = "v0_{}".format(ov(params[0]))
    dims = "v1_{}".format(ov(params[1]))
    shape0 = shape_dic[params[0]]
    shape1 = shape_dic[params[1]]
    t = type_map[type_dic[i.name]]
    tv = type_map[type_dic[params[0]]]
    ts = type_map[type_dic[params[1]]]
    
    if len(shape1) == 0 or (len(shape1) == 1 and shape1[0]==1):
        return impl_sum_1(i, params)
    if tuple(shape0) == tuple(shape_dic[i.name]):
        return impl_sum_0(i, params)
    
    # print(i, len(shape1), shape1, shape0, shape_dic[i.name])
    
    ret += "// init\n "
    ret += "for (int i=0;i<{}; i++) {{".format(shape_size(shape_dic[params[-1]]))
    ret += "*(({}*){} + i) = 0;".format(t, var)
    ret += "}\n"
    ret += "int idx = 0;\n"
    n = len(shape0)
    ret += "long {};".format(",".join(["n_idx_{}={}".format(j, shape0[j] if shape0[j]  else 1) for j in range(len(shape0))]))
    if len(shape1) == 0:
        ret += "int i=0;"
    else:
        ret += "for (int i=0;i<{}; i++) {{".format(shape1[0])
    for j in range(len(shape0)):
        ret += "if ( ((*(({} * ) {} + i)) + {}) % {} == {}) {{".format(ts, dims, len(shape0), len(shape0), j)
        ret += "n_idx_{} = 1;".format(j)
        ret += "}\n"
    if len(shape1) != 0:
        ret += "}"
    for k in range(n):
        ret += "for (int idx_{}=0; idx_{}< {}; idx_{} ++) {{\n".format(k, k, shape0[k] if shape0[k] else 1, k)
    new_idx = "({} * {} + ((idx_{} + n_idx_{}) % n_idx_{}))"
    for j in range(n-1):
        nj = n-1-j
        n_size = "n_idx_{}".format(nj)
        new_idx = new_idx.format("({} * {} + ((idx_{} + n_idx_{}) % n_idx_{}))", n_size, nj, nj, nj)
    new_idx = new_idx.format(0, 1, 0, 0, 0)
    ret += " *(({}*){} + {}) += *(({} *) {} + idx);\n".format(t, var, new_idx, tv, value)
    ret += "idx++;"
    ret += "}" * n
    ret += "}\n"
    return ret


# In[11]:



def def_type_array(otr_dic):
    ret = ""
    for (k1, k2), v in otr_dic.items():
        if len(v) == 0:
            continue
        ret += "typedef {} {}{};\n".format(k1, k1+v.replace("[", "_l_").replace("]", "_r_"), v)
    return ret


def def_call_impl(new_node, type_dic, inputs):
    ret = ""
    for i in new_node:
        if i.name in inputs:
            continue
        params = list(i.input) 
        if i.op != "Unpack":
            params += [ i.name ]
        else:
            params += [ i.name if k==0 else i.name + ":" +str(k) for k in range(i.attr["num"].i) ]
        ret += impl_call(i, params)
    return ret                        



def impl_call(i, params):
    if i.op in impl_call_dic:
        params = [i for i in params if not i.startswith("^")]
        return impl_call_dic[i.op](i, params)
    print(i.op)
    raise NotImplementedError
    return  "void inline {}({}) {{}};".format(ov(i.name) + "_"+i.op, ",".join(["{}* v{}_{}".format(otr(j), it, ov(j)) for it, j in enumerate(params)])) + "\n"


code3 = def_call_impl(new_node, type_dic, input_position_var)


# In[12]:


total_code = "#include\"temple.h\"\n"
total_code += def_type_array(otr_dic)
total_code += code3 +"\n"
total_code += "double my_amp({}) {{\n".format(",".join(["{}& {}".format(otr(j),  ov(j)) for j in input_position_var ]))
total_code += code1 + "\n"
total_code += code2 + "\n"
total_code += "return {}[0];\n}}\n".format(ov(new_node[-1].name))


total_code += """

extern "C" {{
double amp(_pt* ptr){{
    std::cout << "call amp " << std::endl;
    return my_amp({});
}}
}}

""".format(",".join(["*(ptr + {})".format(i) for  i in range(n_input_p)]))

# In[13]:


with open(f"{dir_name}/temple.cpp", "w") as f:
    f.write(total_code)

header_file_str =  """
#include<complex>
#include<iostream>
#include"math.h"

#ifndef TMP_AMP_H
#define TMP_AMP_H

#ifndef DEBUG
#define DEBUG 0
#endif
typedef std::complex<double> complex;
typedef double _pt[1][4];
double my_amp({});

#endif

""".format(",".join(["_pt& p_{}".format(i) for  i in range(n_input_p)]), ",".join(["*(ptr + {})".format(i) for  i in range(n_input_p)]))

with open(f"{dir_name}/temple.h", "w") as f:
    f.write(header_file_str)


testmain_var_def = ""
for i in range(n_input_p):
    testmain_var_def += "    double p_{}[1][4] = {{ {} }};\n".format(i, ",".join([str(a[i][0][j]) for j in range(4)]))

                                                                        
testmain_file_str =  """
#include "temple.h"
#include <iostream>

int main(int argc, char** argv) {{
    {}
    double ret = my_amp({});
    std::cout << ret << std::endl;
}}

""".format(testmain_var_def, ",".join(["p_{}".format(i) for i in range(n_input_p)]))


with open(f"{dir_name}/temple_main.cpp", "w") as f:
    f.write(testmain_file_str)

# In[14]:

subprocess.run(f'g++ {dir_name}/temple.cpp {dir_name}/temple_main.cpp -o {dir_name}/temple_main -I{dir_name} -g -DDEBUG=1'.split(" "))
# get_ipython().system('g++ temple.cpp temple_main.cpp -o temple_main -g -DDEBUG=1')
lines = subprocess.run(f'{dir_name}/temple_main', capture_output=True).stdout.decode()
# get_ipython().system('./temple_main > temple_main.log')


# In[15]:


#with open("b.log", "w") as f:
    #for node, v in zip([i for i in new_node if i.name not in ["p_0", "p_1", "p_3", "p_2"] and ":" not in i.name], all_val):
        #print(node.name, " ".join([str(i) for i in np.reshape(v, (-1,)).tolist()]), file=f)


# In[16]:


with open(f"{dir_name}/temple_main.log", "w") as f:
    f.write(lines)
lines = lines.split("\n")

eval_value = {}
for i in lines:
    val = i.strip().split(" ")
    tmp = [eval(j, {"inf": float("inf"), "nan": float("nan")}) for j in val[1:]]
    eval_value[val[0]] = [complex(j[0], j[1]) if isinstance(j, tuple) else j for j in tmp]
# print(eval_value)


# In[17]:


model_value = {}
for node, v in zip([i for i in new_node if i.name not in input_position_var and ":" not in i.name], all_val):
    model_value[node.name] = np.reshape(v, (-1,)).tolist()


# In[18]:

is_diff = False
for k, v in model_value.items():
    if k in eval_value:
        if len(eval_value[k]) != len(v):
            if "unstack" in k:
                continue
            print(k,"not shape", eval_value[k],v)
            continue
        if not  np.allclose(eval_value[k],v):
            is_diff = True
            print(k, eval_value[k],v)

if is_diff:
    print("model results not true")
else:
    print("success")
