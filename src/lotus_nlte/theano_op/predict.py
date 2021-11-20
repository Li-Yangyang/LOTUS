# -*- coding: utf-8 -*-
import theano

import theano.tensor as tt
import numpy as np

class GenerateMets(theano.Op):
    
    itypes=[tt.dvector]
    otypes=[tt.dscalar]
    
    def __init__(self, mgcog):
        self.mgcog = mgcog

    def perform(self, node, inputs, outputs):
            ews = self.mgcog.obs_ew
            models = self.mgcog.models
            idx_fei = np.where(np.array(self.mgcog.obs_ele) =="FeI")
            theta, = inputs
            
            if self.mgcog.interp_method == "[2-5]":
                generated_mets = []
                for i in range(len(models)):
                    predict_x0, predict_x1, predict_x2, predict_x3 = np.meshgrid(theta[0], theta[1], theta[2], ews[i])
                    predict_x = np.concatenate((predict_x0.reshape(-1, 1), 
                                            predict_x1.reshape(-1, 1),
                                            predict_x2.reshape(-1, 1),
                                            predict_x3.reshape(-1, 1)), 
                                           axis=1)
                    #predict_x_ = poly.fit_transform(predict_x)
                    predict_y = models[i].predict(predict_x)
                    generated_mets.append(predict_y)
                
            if "RBF" in self.mgcog.interp_method:
                generated_mets = []
                for i in range(len(models)):
                    predict_x = np.array([[theta[0], theta[1], theta[2], ews[i]]])
                    predict_y = models[i](predict_x)
                    generated_mets.append(predict_y)
                    
            if self.mgcog.interp_method == "SKIGP":
                import torch
                generated_mets = []
                for i in range(len(models)):
                    predict_x = torch.from_numpy(np.array([[theta[0], theta[1], theta[2], ews[i]]])).to(torch.float)
                    predict_ys = models[i].predict(predict_x)
                    generated_mets.append(predict_ys[0])
                
            generated_mets = np.array(generated_mets)
            outputs[0][0] = np.mean(generated_mets)
