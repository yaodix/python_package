"""
切割细长标注图像 ，暂支持切分为2份
"""

import cv2
import io
import base64
import labelme
import json
import numpy as np
import copy
import os
import shutil
from matplotlib import pyplot as plt

"""
minlen_type: 最小距离的计算方式，0：max，1:min

"""
class AnnoSplit():
    def __init__(self,img_root_path,json_path,
                 split_root_path,label_path , min_len,minlen_type=0,gap = 64):

        self.json_path = json_path
        self.img_path =os.path.join(img_root_path, self._load_json()["imagePath"])
        self.img_name =  self._load_json()["imagePath"]
        self.split_root_path = split_root_path
        self.shapes = self._load_json()["shapes"]
        self.map_name_index = self._get_label_map(label_path)
        self.map_index_name = dict(zip(self.map_name_index.values(),self.map_name_index.keys()))
        self.min_len = min_len
        self.minlen_type = minlen_type
        self.gap = gap
    def _load_json(self):
        '''
        utf-8 与 非utf-8 兼容
        '''
        try:
            data = json.load(open(self.json_path, "r" , encoding='utf-8'))
        except Exception as ex:
            print("no utf-8 decode ")
            data = json.load(open(self.json_path, "r" ))

        return data

    @staticmethod
    def calc_imageDate(img_path):
        data = labelme.LabelFile.load_image_file(img_path)
        image_data = base64.b64encode(data).decode('utf-8')
        return image_data

    def show_annno(self):
        img = cv2.imdecode(np.fromfile(self.img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        colors = [(0,255,0),(0,0,255),(200,100,0),(100,0,200),(100,100,200),(100,100,200),(100,100,200)]
        r,c = img.shape[:-1]
        for hh in self.shapes:
            label = self.map_name_index[hh["label"]]
            # print(label)
            poly_pts = hh["points"]
            arr = np.array(poly_pts,np.int32)
            img = cv2.polylines(img,[arr],True,colors[label-1],3)
        img = cv2.drawMarker(img,(int(c/2),int(r/2)),(255,0,0),markerType= cv2.MARKER_CROSS,markerSize=300,thickness=1)
        return  img

    def create_parts_ver_gap(self):
        '''
        创建2个竖直分割矩形
        :param ver_cnt: 默认切位2段, 
        :param gap : 切分位置的overlap
        :return:
        '''
        if not os.path.exists(self.split_root_path) :
            os.makedirs(self.split_root_path)
        split_name = self.img_name[:-4]
        img_l_path = os.path.join(self.split_root_path,split_name+"_left")
        img_r_path = os.path.join(self.split_root_path,split_name+"_right")
        
        #split img file
        img = cv2.imdecode(np.fromfile(self.img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        r,c = img.shape[:-1]
        y_cent = r_cent = r//2
        x_cent = c_cent = c//2
        split_c_1 = c_cent + self.gap
        split_c_2 = c_cent - self.gap

        img_left = img[:,0:split_c_1,:]
        img_right = img[:,split_c_2:c,:]

        cv2.imencode('.png',  img_left)[1].tofile(img_l_path+".png")
        cv2.imencode('.png',  img_right)[1].tofile(img_r_path+".png")

        #split json file,only split intersection area
        canvs = np.zeros_like(img,np.uint8)
        intersec_cont = {}
        conts_left = {}; conts_mid = {}; conts_right = {}

        for shape in self.shapes:
            label = shape["label"]
            points = shape["points"]
            cont = np.array(points)
            #判断cont所在区域
            if (cont[:,0] <= split_c_1).all():   # tl ,label:conts
                cont_insert = cont-[0,0]      #shift pts
                conts_left = self._insert_dict_list_val(conts_left,label,cont_insert)

            if (cont[:,0]  >= split_c_2).all() : # dr
                cont_insert = cont-[split_c_2,0] 
                conts_right = self._insert_dict_list_val(conts_right,label,cont_insert)
            # 端点在overlap
            if (cont[:,0] >= split_c_2).any() and (cont[:,0] <= split_c_1).any() and ((cont[:,0] < split_c_2).any() or (cont[:,0] > split_c_1).any()) : 
                intersec_cont = self._insert_dict_list_val(intersec_cont,label,cont)
                
            # 端点在overlap两端

            if ((cont[:,0] < split_c_2).any() and (cont[:,0] > split_c_1).any()) : 
                intersec_cont = self._insert_dict_list_val(intersec_cont,label,cont)

        left_inter_conts,right_inter_conts = self._split_intersec_anno_ver(img, intersec_cont)

        #保存对应json文件
        cont_left_all = self._merge_dict(left_inter_conts , conts_left)
        cont_right_all = self._merge_dict(right_inter_conts , conts_right)

        if len(cont_left_all) is 0:
            os.remove(img_l_path+".png")
        else:
            self._save_json_file(img_left , cont_left_all, self.split_root_path,split_name,"_left")

        if len(cont_right_all) is 0:
            os.remove(img_r_path+".png")
        else:
            self._save_json_file(img_right , cont_right_all, self.split_root_path,split_name,"_right")


    def _insert_dict_list_val(self,dict1,key,val):
        if val.__len__() <1:
            pass
        elif dict1 is None or key not in dict1.keys():
            dict1[key] = [val]
        else:
            if not np.array(val ) in np.array( dict1[key]):
                dict1[key].append(val)

        return  dict1

    def _merge_dict(self,dict1,dict2):
        for key1,val1 in dict1.items():
            if key1 not in dict2.keys():
                dict2[key1] = val1
            else:
                dict2[key1].extend(val1)
        return dict2

    def _split_intersec_anno_ver(self, img, intersec_cont):
        left_conts = {}; mid_conts = {}; right_conts = {}
        for key,conts in intersec_cont.items():  #不同图片处理不同类别
            canvs = np.zeros_like(img, np.uint8)
            for cont in conts:
                cont = np.array(cont,np.int32)
                cv2.drawContours(canvs,[cont],0,(255,255,255),-1)
            map_part_conts = self._split_img_cont_ver(canvs)

            left_conts = self._insert_dict_list_val(left_conts,key,map_part_conts["_left"])
            right_conts = self._insert_dict_list_val(right_conts,key,map_part_conts["_right"])

        for k,v in left_conts.items():
            left_conts[k] = v[0]
        for k,v in right_conts.items():
            right_conts[k] = v[0]

        return left_conts, right_conts

    def _split_img_cont_ver(self, canv_img):
        r,c = canv_img.shape[:-1]
        r_cent = r//2
        c_cent = c//2

        img_left = canv_img[:,0:c_cent + self.gap,:]
        img_right = canv_img[:, c_cent - self.gap : c,:]

        map_img = {"_left":img_left,"_right":img_right}
        conts_left = [];    conts_right = []
        map_conts = {"_left":conts_left, "_right":conts_right }

        for pos_name,patch in map_img.items():
            shapes = []
            shape={}
            if patch.shape[1] == 0:
                continue
            part_img = cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY)
            _,part_img_bin = cv2.threshold(part_img, 0, 255, cv2.THRESH_BINARY)
            conts,_ = cv2.findContours(part_img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
            #判断conts 删除外接矩形长度小于50
            filter_conts = []
            for cont in conts:
                cont = np.squeeze(cont)
                if len(cont) >2:
                    bbox = cv2.minAreaRect(cont)

                    if self.minlen_type == 1:
                        max_len =min(bbox[1][0],bbox[1][1])
                    elif self.minlen_type == 0:
                        max_len =max(bbox[1][0],bbox[1][1])

                    if max_len >  self.min_len:
                        map_conts[pos_name].append(cont)

        return map_conts
    def _save_json_file(self,patch,conts,split_root_path,split_name,pos_suffix):
        shape ={}
        shapes =[]
        for label,cont in conts.items():
            shape["label"] = label
            for c in cont:
                arr_c = np.array(c)
                list_c = arr_c.tolist()
                shape["points"] = list_c
                shape["group_id"] = None
                shape["shape_type"] = "polygon"
                shape["flags"] = {}
                shapes.append(copy.deepcopy(shape))

        image_path = split_name + pos_suffix + ".png"
        # image_data = self.calc_imageDate(os.path.join(split_root_path,split_name+pos_name)+".png")
        image_data = labelme.LabelFile.load_image_file(os.path.join(split_root_path, split_name + pos_suffix) + ".png")
        image_height = patch.shape[0]
        image_width = patch.shape[1]
        file_name = os.path.join(self.split_root_path, split_name + pos_suffix + ".json")
        lf = labelme.LabelFile()

        lf.save(filename=file_name, shapes=shapes, imagePath=image_path, imageHeight=image_height,
                imageWidth=image_width, imageData=image_data)
        print(f"save json {split_name + pos_suffix}")

    def _get_split_json(self,canv_img,pt,split_root_path,split_name):
        r,c = canv_img.shape[:-1]
        r_cent = pt[1]
        c_cent = pt[0]
        img_tl = canv_img[0:r_cent,0:c_cent,:]
        img_tr = canv_img[0:r_cent,c_cent:c,:]
        img_dr = canv_img[r_cent:r,c_cent:c,:]
        img_dl = canv_img[r_cent:r,0:c_cent,:]
        map_img = {"_dr":img_dr,"_dl":img_dl,"_tr":img_tr,"_tl":img_tl}
        for pos_name,patch in map_img.items():
            # print(f"img  {pos_name}")
            shapes = []
            shape={}
            part_img = cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY)
            _,part_img_bin = cv2.threshold(part_img, 0, 255, cv2.THRESH_BINARY)
            # conts,_ = cv2.findContours(part_img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            conts,_ = cv2.findContours(part_img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
            #没有标注区域删除子图片
            part_img_path = os.path.join(split_root_path,split_name+pos_name)+".png"

            #判断conts 删除外接矩形长度小于50
            filter_conts = []
            for cont in conts:
                cont = np.squeeze(cont)
                bbox = cv2.minAreaRect(cont)
                len =max(bbox[1][0],bbox[1][1])
                if len >  self.min_len:
                    filter_conts.append(cont)
            if conts.__len__() < 1 or filter_conts.__len__() <1:
                os.remove(part_img_path)
                continue
            for cont in filter_conts:
                cont = np.squeeze(cont)
                scalar = self._get_nonzero_pt_neigbor(part_img,cont[0],2)
                shape["label"] = self.map_index_name[scalar]
                list_cont =( cont.tolist())
                shape["points"] = list_cont
                shape["group_id"] = None
                shape["shape_type"] = "polygon"
                shape["flags"] = {}
                shapes.append(copy.deepcopy(shape))

            image_path = split_name+pos_name+".png"
            image_data =  labelme.LabelFile.load_image_file(os.path.join(split_root_path,split_name+pos_name)+".png")
            image_height = patch.shape[0]
            image_width = patch.shape[1]
            file_name = os.path.join(self.split_root_path,split_name+pos_name+".json")
            lf =labelme.LabelFile()
            lf.save(filename=file_name,shapes = shapes,imagePath = image_path,imageHeight= image_height,imageWidth=image_width,imageData= image_data)
            print(f"save json {split_name+pos_name}")
            # break
        print(f"{self.img_name} split done!")

    def _get_nonzero_pt_neigbor(self,img,pt,half_len):
        if img.shape[-1] >1:
            img_gray = img
        else:
            img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # assume all pt inside image
        c_cent =pt[0]
        r_cent =pt[1]
        h,w = img.shape
        img_patch = img_gray[max(r_cent-half_len,0):min(r_cent+half_len,h),
                             max(c_cent-half_len,0):min(c_cent+half_len,w)]
        val = img_patch[img_patch>0]
        return val[0]

    def _get_label_map(self,labentxt):
        map_label={}
        i=1
        for obj in open(labentxt):
            if obj[0] is not '_':
                map_label[obj.strip()] = i
                i += 1
        return map_label



from glob import glob
from tqdm import tqdm
import shutil

if __name__ == "__main__":

    root_path = r"D:\侧面非划痕缺陷样本\DLData\LFPFL-frame\Version"

    save_dir_list = [
        r"D:\侧面非划痕缺陷样本\DLData\LFPFL-frame\Version\A1-V0.4-gray-train-liuyao\A1-KLF-LF-PLF-DQ-exp",
        r"D:\侧面非划痕缺陷样本\DLData\LFPFL-frame\Version\A1-V0.4-gray-train-liuyao\A1-LF-add-exp",
        r"D:\侧面非划痕缺陷样本\DLData\LFPFL-frame\Version\A1-V0.4-gray-train-liuyao\A1-LF-exp",
        r"D:\侧面非划痕缺陷样本\DLData\LFPFL-frame\Version\A1-V0.4-gray-train-liuyao\A1-PFL-add-exp",
        r"D:\侧面非划痕缺陷样本\DLData\LFPFL-frame\Version\A1-V0.4-gray-train-liuyao\A1-PFL-exp",
        r"D:\侧面非划痕缺陷样本\DLData\LFPFL-frame\Version\A1-V0.4-gray-train-liuyao\A1-TXQS-exp",
    ]

    json_dir_list =[  
    # r"E:\5_Snapbox\8_DLData\LFPFL\A1_part",
    # r"D:\侧面非划痕缺陷样本\DLData\LFPFL-frame\Version\L1-V0.5\L1-LF-DQ-exp",
    # r"D:\侧面非划痕缺陷样本\DLData\LFPFL-frame\Version\L1-V0.5\L1-LF-PFL-CE-DQ-exp",
    # r"D:\侧面非划痕缺陷样本\DLData\LFPFL-frame\Version\L1-V0.5\L1-PFL-DQ-exp",
    # r"D:\侧面非划痕缺陷样本\DLData\LFPFL-frame\Version\A1-V0.4-gray\L1-TXQS-exp",
        r"D:\侧面非划痕缺陷样本\DLData\LFPFL-frame\Version\A1-V0.4-gray\A1-KLF-LF-PLF-DQ-exp",
        r"D:\侧面非划痕缺陷样本\DLData\LFPFL-frame\Version\A1-V0.4-gray\A1-LF-add-exp",
        r"D:\侧面非划痕缺陷样本\DLData\LFPFL-frame\Version\A1-V0.4-gray\A1-LF-exp",
        r"D:\侧面非划痕缺陷样本\DLData\LFPFL-frame\Version\A1-V0.4-gray\A1-PFL-add-exp",
        r"D:\侧面非划痕缺陷样本\DLData\LFPFL-frame\Version\A1-V0.4-gray\A1-PFL-exp",
        r"D:\侧面非划痕缺陷样本\DLData\LFPFL-frame\Version\A1-V0.4-gray\A1-TXQS-exp",

        ]
    
    img_file_ext ="png"
    
    defect_min_len = 30

    labeltxt_path = r'D:\侧面非划痕缺陷样本\DLData\LFPFL-frame\Version\A1-V0.4-gray\label.txt'

    for save_dir in save_dir_list:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        else:
            shutil.rmtree(save_dir)
            os.makedirs(save_dir)
        
    split_total =0
    move_total =0
    for i, json_dir in tqdm(enumerate( json_dir_list)):
        save_dir = save_dir_list[i]

        glob_path = os.path.join(json_dir,"*.json")
        img_root_dir = json_dir
        json_files =glob(glob_path)

        # filter 
        js_files = json_files
        all_total = all_total+js_files.__len__()
        for j in tqdm(js_files):
            # if not "284532-" in j:
                # continue
            _, json_name = os.path.split(j)
            img_name = json_name.replace("json",img_file_ext)  
            if "-C-" in img_name or "-E-" in img_name:
                img_src_path = j.replace("json",img_file_ext)
                img_dst_path = os.path.join(save_dir,img_name)
                json_dst_path = os.path.join(save_dir,json_name)
                shutil.copy(img_src_path, img_dst_path)
                shutil.copy(j, json_dst_path)
                print(f" move {img_name}")
                move_total +=1
                continue
                        
            anno_split = AnnoSplit(img_root_dir,j,save_dir,labeltxt_path,min_len =defect_min_len,minlen_type=0,gap=64 )            
            anno_split.create_parts_ver_gap()
            split_total += 1            
            print(f"img splited")
            # break

    print(f"split total is {split_total}, move total is {move_total}, all_total is {all_total}")

