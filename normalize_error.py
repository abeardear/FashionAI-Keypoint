#encoding:utf-8
'''
根据valid.csv pred.csv评价效果
两个csv文件格式与提交格式相同
'''
import pandas as pd
import math
import numpy as np

def distance(x1,y1,x2,y2):
    l1 = math.pow((x1-x2),2)
    l2 = math.pow((y1-y2),2)
    return math.sqrt(l1+l2)

target = pd.read_csv('valid.csv')
# target = pd.read_csv('testb_label.csv')
# pred = pd.read_csv('fusemodel/predblouseargmax.csv')
pred = pd.read_csv('pred-merge.csv')
# pred = pd.read_csv('pred-cpn.csv')
# target = pd.DataFrame([['0000001.jpg','trousers','430_284_0','713_303_0','560_537_1','560_626_1','361_588_1','573_622_1','-1_-1_-1'],
#     ['0000002.jpg','trousers','359_301_1','464_297_1','417_403_1','340_669_1','308_658_1','456_713_1','491_714_1']],
#     columns=['image_id','image_category','waistband_left','waistband_right','crotch','bottom_left_in','bottom_left_out','bottom_right_in','bottom_right_out'])

# pred = pd.DataFrame([['0000001.jpg','trousers','430_294_0','713_323_0','560_567_1','560_666_1','361_638_1','573_682_1','123_345_1'],
#     ['0000002.jpg','trousers','359_311_1','464_317_1','417_433_1','340_709_1','308_708_1','456_773_1','491_784_1']],
#     columns=['image_id','image_category','waistband_left','waistband_right','crotch','bottom_left_in','bottom_left_out','bottom_right_in','bottom_right_out'])

numAnno = len(target)
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

NEsum = 0
count_vis = 0
count = 0
diffMean = np.zeros(2)
j = 0
for i in range(numAnno):
    anno = target.ix[i]
    class_name = anno['image_category']
    # if class_name != 'blouse': #如果只测试一种类别,把这里取消注释
    #     continue
    result = pred.ix[i]
    # print(anno['image_id'])
    assert anno['image_id'] == result['image_id'] #出错一般是编号没有对其,检查i j
    # print(anno['image_id'])
    if class_name == 'trousers' or class_name == 'skirt':
        # print('waistband')
        x1,y1,_ = anno['waistband_left'].split('_')
        x2,y2,_ = anno['waistband_right'].split('_')
    if class_name == 'blouse' or class_name == 'outwear' or class_name == 'dress':
        # print('armpit')
        x1,y1,_ = anno['armpit_left'].split('_')
        x2,y2,_ = anno['armpit_right'].split('_')

    x1,y1,x2,y2 = float(x1), float(y1), float(x2), float(y2)
    normalize = distance(x1,y1,x2,y2)
    if(normalize==0):
        print('WTF')
        normalize = 100.0
    pts = []
    pts_pred = []
    visiable_count = 0
    for part in part_name[class_name]:
        xt,yt,visiable = anno[part].split('_')
        visiable = int(visiable)
        visiable_count += visiable
        xp,yp,_ = result[part].split('_')
        if visiable == 1:
            pts.append((float(xt),float(yt)))
            # pts_pred.append((float(xp)-0,float(yp)-2))
            pts_pred.append((float(xp)+0.,float(yp)-0.)) #弥补统计偏差x_offset后的效果?
    if visiable_count==0:
        j += 1
        continue
    pts = np.array(pts)
    pts_pred = np.array(pts_pred)
    # pts_pred = np.around(pts_pred).astype(np.int32)
    diff_mean = pts-pts_pred
    diffMean += np.mean(diff_mean,0)#统计偏差x_offset,y_offset用的
    diff = (pts-pts_pred) ** 2
    # print(diff)
    diss = np.sqrt( np.sum(diff,1) )
    total_diss = np.sum(diss)
    NE = total_diss/normalize#/len(pts)
    NEsum += NE
    count_vis += len(pts)
    count += 1
    j += 1
NEsum = NEsum/count_vis
diffMean = diffMean/count
print('NE: {}'.format(NEsum))
print(diffMean)
