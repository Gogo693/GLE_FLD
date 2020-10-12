import torch
import torch.nn as nn
import numpy as np

import cv2
import matplotlib.pyplot as plt

import json

def drawKeyPts(im,keyp,col,th):
    for curKey in keyp:
        x=np.int(curKey[0])
        y=np.int(curKey[1])
        size = 1
        cv2.circle(im,(x,y),size, col,thickness=th, lineType=8, shift=0)
        #print(x, y) 
    plt.imshow(im)
    plt.show()
    plt.close() 
    return im 

def drawImg(im,keyp, name):
    plt.imshow(im)

    for curKey in keyp:
        x=np.int(curKey[0])
        y=np.int(curKey[1])
        plt.scatter(x, y, color = 'r')
        #print(x, y) 

    #plt.show()   
    print(name)
    plt.savefig('./results/' + name.split('/')[2])
    plt.clf()

def cal_loss(sample, output):
    batch_size, _, pred_w, pred_h = sample['image'].size()

    lm_size = int(output['lm_pos_map'].size(2))
    visibility = sample['landmark_vis']
    vis_mask = torch.cat([visibility.reshape(batch_size* 8, -1)] * lm_size * lm_size, dim=1).float()
    lm_map_gt = sample['landmark_map%d' % lm_size].reshape(batch_size * 8, -1)
    lm_pos_map = output['lm_pos_map']
    lm_map_pred =lm_pos_map.reshape(batch_size * 8, -1)

    loss = torch.pow(vis_mask * (lm_map_pred - lm_map_gt), 2).mean()

    return loss

class Evaluator(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.lm_vis_count_all = np.array([0.] * 8)
        self.lm_dist_all = np.array([0.] * 8)

    def add(self, output, sample, img):
        landmark_vis_count = sample['landmark_vis'].cpu().numpy().sum(axis=0)
        landmark_vis_float = torch.unsqueeze(sample['landmark_vis'].float(), dim=2)
        landmark_vis_float = torch.cat([landmark_vis_float, landmark_vis_float], dim=2).cpu().detach().numpy()

        lm_pos_map = output['lm_pos_map']

        batch_size, _, pred_h, pred_w = lm_pos_map.size()
        lm_pos_reshaped = lm_pos_map.reshape(batch_size, 8, -1)

        lm_pos_y, lm_pos_x = np.unravel_index(torch.argmax(lm_pos_reshaped, dim=2).cpu().numpy(), (pred_h, pred_w))
        #print(lm_pos_y, lm_pos_x)
        #lm_pos_output = np.stack([lm_pos_x / (pred_w - 1), lm_pos_y / (pred_h - 1)], axis=2)
        lm_pos_output = np.stack([lm_pos_x, lm_pos_y], axis=2)
        #print(len(lm_pos_output))
        #print(len(lm_pos_output[0]))
        #print(lm_pos_output[0])

        x_v = lm_pos_x[0]
        y_v = lm_pos_y[0]
        #print(x_v)
        #print(len(y_v))
        keyp = []
        for i, v in enumerate(x_v):
            keyp.append([x_v[i], y_v[i]])
        #print(keyp)
        
        imag = np.zeros([224,224,3],dtype=np.uint8)
        imag.fill(255)
        #imag = io.imread('./img/'+ sample['image_name'])

        '''
        from skimage import io, transform

        bbox_crop = BBoxCrop()
        rescale224square = Rescale((224, 224))

        imag = io.imread('./img/' + image_name[0])

        transform.resize(imag, (224, 224))
        '''
        
        
        for i, single_image in enumerate(img):
            #drawImg(single_image, lm_pos_output[i], str(sample['image_name'][i]))
            #drawImg(single_image, sample['landmark_pos'][i].cpu())
            data = {}
            data['landmarks'] = []
            for cord in lm_pos_output[i]:
                print(cord)
                data['landmarks'].append(float(cord[0]))
                data['landmarks'].append(float(cord[1]))
            
            with open('./landmarks/' + str(sample['image_name'][i].split('/')[2]'.json', 'w') as outfile:
                json.dump(data, outfile) 

            #print(asd)
        

        #img = np.transpose(img[0], (-1, 0, 1))
        #print(img.shape)
        #print((np.transpose(np.array(img[0]), (-1, 0, 1))).shape)
        #drawKeyPts(np.transpose(np.uint8(img[0]), (-1, 0, 1)), lm_pos_output[0], (0,255,0),5)
        #drawKeyPts(np.transpose(np.uint8(img[0]), (-1, 0, 1)), sample['landmark_pos'][0].cpu(), (0,255,0),5)

        landmark_dist = np.sum(np.sqrt(np.sum(np.square(
            landmark_vis_float * lm_pos_output - landmark_vis_float * sample['landmark_pos_normalized'].cpu().numpy(),
        ), axis=2)), axis=0)

        self.lm_vis_count_all += landmark_vis_count
        self.lm_dist_all += landmark_dist

    def evaluate(self):
        lm_dist = self.lm_dist_all / self.lm_vis_count_all
        lm_dist_all = (self.lm_dist_all / self.lm_vis_count_all).mean()

        return {'lm_dist' : lm_dist,
                'lm_dist_all' : lm_dist_all}


