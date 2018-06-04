import pandas as pd
import math
import numpy as np

pred = pd.read_csv('argmax-cpnv2-crop.csv')
pred_cpn = pd.read_csv('argmax-cpnv2.csv')

numAnno = len(pred)

blouse_part = ['neckline_left','neckline_right','center_front','shoulder_left','shoulder_right',
    'armpit_left','armpit_right','cuff_left_in','cuff_left_out','cuff_right_in',
    'cuff_right_out','top_hem_left','top_hem_right']
trousers_part = ['waistband_left','waistband_right','crotch','bottom_left_in','bottom_left_out','bottom_right_in','bottom_right_out']

outwear_part = ['neckline_left','neckline_right','shoulder_left','shoulder_right','armpit_left','armpit_right',
    'waistline_left','waistline_right','cuff_left_in','cuff_left_out','cuff_right_in','cuff_right_out','top_hem_left','top_hem_right']

skirt_part = ['waistband_left','waistband_right','hemline_left','hemline_right']

dress_part = ['neckline_left','neckline_right','shoulder_left','shoulder_right','center_front','armpit_left','armpit_right',
    'waistline_left','waistline_right','cuff_left_in','cuff_left_out','cuff_right_in','cuff_right_out','hemline_left','hemline_right']

part_name = {'trousers':trousers_part,'blouse':blouse_part,'outwear':outwear_part,'skirt':skirt_part,'dress':dress_part}

for i in range(numAnno):
    anno = pred.ix[i]
    anno_cpn = pred_cpn.ix[i]
    image_id = anno['image_id']
    image_id2 = anno_cpn['image_id']
    # print(image_id,image_id2)
    assert image_id==image_id2
    class_name = anno['image_category']
    
    for part in part_name[class_name]:
        xt,yt,visiable = anno[part].split('_')
        xcpn,ycpn,visiable_cpn = anno_cpn[part].split('_')
        visiable = int(visiable)
        xt,yt,xcpn,ycpn = float(xt), float(yt), float(xcpn), float(ycpn)
        if visiable==1:
            x = int(round((0.75*xt+0.25*xcpn)))
            y = int(round((0.75*yt+0.25*ycpn)))
            anno[part] = str(x)+'_'+str(y)+'_1'

pred.to_csv('pred-merge.csv',index=False)