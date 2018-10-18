import os
import os.path as osp
import mxnet as mx
import json
import sys
import numpy as np
import copy 

import fresnet

def merge_bn(args, auxs, conv_name, bn_prefix):
    conv_weights = args[conv_name+"_weight"].asnumpy()
    gamma = args[bn_prefix+"_gamma"].asnumpy()
    beta = args[bn_prefix+"_beta"].asnumpy()
    # print('conv_weights.shape={}'.format(conv_weights.shape)) 
    mean = auxs[bn_prefix+"_moving_mean"].asnumpy()
    variance = auxs[bn_prefix+"_moving_var"].asnumpy()
    channels = conv_weights.shape[0]
    epsilon = 2e-5
    rstd = 1. / np.sqrt(variance + epsilon)
    rstd = rstd.reshape((channels, 1, 1, 1))
    gamma = gamma.reshape((channels, 1, 1, 1))
    beta = beta.reshape((channels, 1, 1, 1))
    # bias = bias.reshape((channels, 1, 1, 1))
    mean = mean.reshape((channels, 1, 1, 1))

    new_weights = conv_weights * gamma * rstd
    # new_bias = (bias - mean) * rstd * gamma  + beta
    new_bias = ( - mean) * rstd * gamma  + beta

    new_bias = new_bias.reshape((channels,))

    args[conv_name+"_weight"] = mx.nd.array(new_weights)
    args[conv_name+"_bias"] = mx.nd.array(new_bias)

    # delete 
    args.pop(bn_prefix+"_gamma")
    args.pop(bn_prefix+"_beta")
    auxs.pop(bn_prefix+"_moving_mean")
    auxs.pop(bn_prefix+"_moving_var")

if __name__ == '__main__':
    prefix='r50_128d/model-r50-128d-slim'
    epoch=0
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix,epoch)

    # mxnet_symbol=json.loads(sym.tojson())

    conv_names=['conv0','stage1_unit1_conv1','stage1_unit1_conv2','stage1_unit1_conv1sc','stage1_unit2_conv1','stage1_unit2_conv2','stage1_unit3_conv1','stage1_unit3_conv2','stage2_unit1_conv1','stage2_unit1_conv2','stage2_unit1_conv1sc','stage2_unit2_conv1','stage2_unit2_conv2','stage2_unit3_conv1','stage2_unit3_conv2','stage2_unit4_conv1','stage2_unit4_conv2','stage3_unit1_conv1','stage3_unit1_conv2','stage3_unit1_conv1sc','stage3_unit2_conv1','stage3_unit2_conv2','stage3_unit3_conv1','stage3_unit3_conv2','stage3_unit4_conv1','stage3_unit4_conv2','stage3_unit5_conv1','stage3_unit5_conv2','stage3_unit6_conv1','stage3_unit6_conv2','stage3_unit7_conv1','stage3_unit7_conv2','stage3_unit8_conv1','stage3_unit8_conv2','stage3_unit9_conv1','stage3_unit9_conv2','stage3_unit10_conv1','stage3_unit10_conv2','stage3_unit11_conv1','stage3_unit11_conv2','stage3_unit12_conv1','stage3_unit12_conv2','stage3_unit13_conv1','stage3_unit13_conv2','stage3_unit14_conv1','stage3_unit14_conv2','stage4_unit1_conv1','stage4_unit1_conv2','stage4_unit1_conv1sc','stage4_unit2_conv1','stage4_unit2_conv2','stage4_unit3_conv1','stage4_unit3_conv2']

    bn_prefixes=['bn0','stage1_unit1_bn2','stage1_unit1_bn3','stage1_unit1_sc','stage1_unit2_bn2','stage1_unit2_bn3','stage1_unit3_bn2','stage1_unit3_bn3','stage2_unit1_bn2','stage2_unit1_bn3','stage2_unit1_sc','stage2_unit2_bn2','stage2_unit2_bn3','stage2_unit3_bn2','stage2_unit3_bn3','stage2_unit4_bn2','stage2_unit4_bn3','stage3_unit1_bn2','stage3_unit1_bn3','stage3_unit1_sc','stage3_unit2_bn2','stage3_unit2_bn3','stage3_unit3_bn2','stage3_unit3_bn3','stage3_unit4_bn2','stage3_unit4_bn3','stage3_unit5_bn2','stage3_unit5_bn3','stage3_unit6_bn2','stage3_unit6_bn3','stage3_unit7_bn2','stage3_unit7_bn3','stage3_unit8_bn2','stage3_unit8_bn3','stage3_unit9_bn2','stage3_unit9_bn3','stage3_unit10_bn2','stage3_unit10_bn3','stage3_unit11_bn2','stage3_unit11_bn3','stage3_unit12_bn2','stage3_unit12_bn3','stage3_unit13_bn2','stage3_unit13_bn3','stage3_unit14_bn2','stage3_unit14_bn3','stage4_unit1_bn2','stage4_unit1_bn3','stage4_unit1_sc','stage4_unit2_bn2','stage4_unit2_bn3','stage4_unit3_bn2','stage4_unit3_bn3']

    assert(len(conv_names)==len(bn_prefixes))
    for i in xrange(len(conv_names)):
        conv_name = conv_names[i]
        bn_prefix = bn_prefixes[i]
        merge_bn(arg_params, aux_params, conv_name, bn_prefix)

    emb_size=128
    num_layers=50
    version_se=0
    version_input=1
    version_output='E'
    version_unit=3
    version_act='prelu'
    nobn_sym=fresnet.get_symbol(emb_size, num_layers, 
                version_se=version_se, version_input=version_input, 
                version_output=version_output, version_unit=version_unit,
                version_act=version_act)

    mx.model.save_checkpoint('mergebn_test',0,nobn_sym,arg_params, aux_params)
  
