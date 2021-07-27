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
from numpy.testing._private.utils import print_assert_equal

class AnnoSplit():

    # gap : 切分图像预测时，重叠区域，用于避免预测断裂
    def __init__(self,img_root_dir,json_path,
                split_root_dir,label_path ,gap, min_len):
        self.json_path = json_path
        self.img_path =os.path.join(img_root_dir, self._load_json()["imagePath"])
        self.img_name =  self._load_json()["imagePath"]
        self.split_root_dir = split_root_dir
        self.shapes = self._load_json()["shapes"]
        self.map_name_index = self._get_label_map(label_path)
        self.map_index_name = dict(zip(self.map_name_index.values(),self.map_name_index.keys()))
        self.gap= gap
        self.min_len = min_len
    def _load_json(self):
        '''
        utf-8 与 非utf-8 兼容
        :return:
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

    #split img to 4 part and create anno
    # 存在错误
    def create_parts(self,pt):
        if not os.path.exists(self.split_root_dir) :
            os.makedirs(self.split_root_dir)
        split_name = self.img_name[:-4]
        img_tl_path = os.path.join(self.split_root_dir,split_name+"_tl")
        img_tr_path = os.path.join(self.split_root_dir,split_name+"_tr")
        img_dr_path = os.path.join(self.split_root_dir,split_name+"_dr")
        img_dl_path = os.path.join(self.split_root_dir,split_name+"_dl")
        #split img file
        img = cv2.imdecode(np.fromfile(self.img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        r,c = img.shape[:-1]
        y_cent = r_cent = pt[1]
        x_cent = c_cent = pt[0]
        img_tl = img[0:r_cent+self.gap, 0:c_cent+self.gap,:]
        img_tr = img[0:r_cent+self.gap,c_cent-self.gap:c,:]
        img_dr = img[r_cent-self.gap:r,c_cent-self.gap:c,:]
        img_dl = img[r_cent-self.gap:r,0:c_cent+self.gap,:]
        cv2.imencode('.png', img_tl)[1].tofile(img_tl_path+".png")
        cv2.imencode('.png', img_tr)[1].tofile(img_tr_path+".png")
        cv2.imencode('.png', img_dr)[1].tofile(img_dr_path+".png")
        cv2.imencode('.png', img_dl)[1].tofile(img_dl_path+".png")

        #split json file,only split intersection area
        canvs = np.zeros_like(img,np.uint8)
        intersec_cont = {}
        conts_tl = {}; conts_tr = {};conts_dr = {};conts_dl = {}

        for shape in self.shapes:
            label = shape["label"]
            points = shape["points"]
            cont = np.array(points)
            #判断cont所在区域
            if (cont[:,0] <= x_cent + self.gap).all() and (cont[:,1] <=y_cent+self.gap).all():   # tl ,label:conts
                cont = cont-[0,0]      #shift pts
                conts_tl = self._insert_dict_list_val(conts_tl,label,cont)
            elif (cont[:,0] >= x_cent-self.gap).all() and (cont[:,1] <= y_cent+self.gap).all():  # tr
                cont = cont-[x_cent - self.gap,0]
                conts_tr = self._insert_dict_list_val(conts_tr,label,cont)
            elif (cont[:,0]  >= x_cent - self.gap).all() and (cont[:,1] >= y_cent-self.gap).all(): # dr
                cont = cont-[x_cent-self.gap,y_cent-self.gap]
                conts_dr = self._insert_dict_list_val(conts_dr,label,cont)
            elif (cont[:,0]  <= x_cent+self.gap).all() and (cont[:,1] >= y_cent-self.gap).all():  # dl ,label:conts
                cont = cont-[0,y_cent-self.gap]                              #shift pts
                conts_dl = self._insert_dict_list_val(conts_dl,label,cont)
            else:  # 落在切分区,gap不等于0 时，错误
                intersec_cont = self._insert_dict_list_val(intersec_cont,label,cont)
        tl_inter_conts,tr_inter_conts,dr_inter_conts,dl_inter_conts = self._split_intersec_anno(img,pt,intersec_cont)

        #保存对应json文件
        cont_tl_all = self._merge_dict(tl_inter_conts,conts_tl)
        cont_tr_all = self._merge_dict(tr_inter_conts,conts_tr)
        cont_dr_all = self._merge_dict(dr_inter_conts,conts_dr)
        cont_dl_all = self._merge_dict(dl_inter_conts,conts_dl)
        if len(cont_tl_all) is 0:
            os.remove(img_tl_path+".png")
        else:
            self._save_json_file(img_tl,cont_tl_all,split_name,"_tl")
        if len(cont_tr_all) is 0:
            os.remove(img_tr_path+".png")
        else:
            self._save_json_file(img_tr,cont_tr_all,split_name,"_tr")

        if len(cont_dr_all) is 0:
            os.remove(img_dr_path+".png")
        else:
            self._save_json_file(img_dr,cont_dr_all,split_name,"_dr")
        if len(cont_dl_all) is 0:
            os.remove(img_dl_path+".png")
        else:
            self._save_json_file(img_dl,cont_dl_all,split_name,"_dl")

    
    def _cont_part_in_rect(self,cont, rect):
        # 判断cont 是否有点在rect中
        # rect[tl_x,tl_y,dr_x,dr_y]
        in_rect = False
        for pt in cont:
            if rect[0] < pt[0] and  pt[0] < rect[2] and  rect[1] < pt[1] and  pt[1] < rect[3]:
                in_rect = True
                # print(f"{pt}")

        return in_rect

# gap > 0 的切图
    def create_parts_gap(self,pt):
        if not os.path.exists(self.split_root_dir) :
            os.makedirs(self.split_root_dir)
        split_name = self.img_name[:-4]
        img_tl_path = os.path.join(self.split_root_dir,split_name+"_tl")
        img_tr_path = os.path.join(self.split_root_dir,split_name+"_tr")
        img_dr_path = os.path.join(self.split_root_dir,split_name+"_dr")
        img_dl_path = os.path.join(self.split_root_dir,split_name+"_dl")
        #split img file
        img = cv2.imdecode(np.fromfile(self.img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        r,c = y_end, x_end = img.shape[:-1]
        y_cent = r_cent = pt[1]
        x_cent = c_cent = pt[0]
        img_tl = img[0:r_cent+self.gap, 0:c_cent+self.gap,:]
        img_tr = img[0:r_cent+self.gap,c_cent-self.gap:c,:]
        img_dr = img[r_cent-self.gap:r,c_cent-self.gap:c,:]
        img_dl = img[r_cent-self.gap:r,0:c_cent+self.gap,:]
        cv2.imencode('.png', img_tl)[1].tofile(img_tl_path+".png")
        cv2.imencode('.png', img_tr)[1].tofile(img_tr_path+".png")
        cv2.imencode('.png', img_dr)[1].tofile(img_dr_path+".png")
        cv2.imencode('.png', img_dl)[1].tofile(img_dl_path+".png")

        #split json file,only split intersection area
        canvs = np.zeros_like(img,np.uint8)
        intersec_cont = {}
        conts_tl = {}; conts_tr = {};conts_dr = {};conts_dl = {}
        conts_tl_split = {}; conts_tr_split = {};conts_dr_split = {}; conts_dl_split = {}

        for shape in self.shapes:
            label = shape["label"]
            points = shape["points"]
            cont = np.array(points)
            #判断cont所在区域
            rect_tl = (0, 0, x_cent + self.gap,y_cent+self.gap)
            rect_tr = (x_cent - self.gap, 0, x_end, y_cent+self.gap)
            rect_dr = (x_cent - self.gap, y_cent-self.gap, x_end,y_end)
            rect_dl = (0 , y_cent-self.gap, x_cent+self.gap, y_end)

            if (cont[:,0] <= x_cent + self.gap).all() and (cont[:,1] <=y_cent+self.gap).all():   # tl ,label:conts
                cur_cont = cont-[0,0]      #shift pts
                conts_tl = self._insert_dict_list_val(conts_tl,label,cur_cont)
            elif self._cont_part_in_rect(cont, rect_tl):  # 有点落在该区域
                conts_tl_split = self._insert_dict_list_val(conts_tl_split,label,cont)

            if (cont[:,0] >= x_cent-self.gap).all() and (cont[:,1] <= y_cent+self.gap).all():  # tr
                cur_cont = cont-[x_cent - self.gap,0]
                conts_tr = self._insert_dict_list_val(conts_tr,label,cur_cont)
            elif self._cont_part_in_rect(cont, rect_tr):  # 有点落在该区域
                conts_tr_split = self._insert_dict_list_val(conts_tr_split,label,cont)

            if (cont[:,0]  >= x_cent - self.gap).all() and (cont[:,1] >= y_cent-self.gap).all(): # dr
                cur_cont = cont-[x_cent-self.gap,y_cent-self.gap]
                conts_dr = self._insert_dict_list_val(conts_dr,label,cur_cont)
            elif self._cont_part_in_rect(cont, rect_dr):  # 有点落在该区域
                conts_dr_split = self._insert_dict_list_val(conts_dr_split,label,cont)

            if (cont[:,0]  <= x_cent+self.gap).all() and (cont[:,1] >= y_cent-self.gap).all():  # dl ,label:conts
                cur_cont = cont-[0,y_cent-self.gap]                              #shift pts
                conts_dl = self._insert_dict_list_val(conts_dl,label,cur_cont)
            elif self._cont_part_in_rect(cont, rect_dl):  # 有点落在该区域
                conts_dl_split = self._insert_dict_list_val(conts_dl_split,label,cont)

        # tl_inter_conts,tr_inter_conts,dr_inter_conts,dl_inter_conts = self._split_intersec_anno(img,pt,intersec_cont)
        tl_inter_conts = self._split_intersec_anno_part(img, conts_tl_split, rect_tl , "_tl")
        tr_inter_conts = self._split_intersec_anno_part(img, conts_tr_split, rect_tr,  "_tr")
        dr_inter_conts = self._split_intersec_anno_part(img, conts_dr_split, rect_dr, "_dr")
        dl_inter_conts = self._split_intersec_anno_part(img, conts_dl_split, rect_dl, "_dl")
        #保存对应json文件
        cont_tl_all = self._merge_dict(tl_inter_conts,conts_tl)
        cont_tr_all = self._merge_dict(tr_inter_conts,conts_tr)
        cont_dr_all = self._merge_dict(dr_inter_conts,conts_dr)
        cont_dl_all = self._merge_dict(dl_inter_conts,conts_dl)
        if len(cont_tl_all) is 0:
            os.remove(img_tl_path+".png")
        else:
            self._save_json_file(img_tl,cont_tl_all,split_name,"_tl")
        if len(cont_tr_all) is 0:
            os.remove(img_tr_path+".png")
        else:
            self._save_json_file(img_tr,cont_tr_all,split_name,"_tr")

        if len(cont_dr_all) is 0:
            os.remove(img_dr_path+".png")
        else:
            self._save_json_file(img_dr,cont_dr_all,split_name,"_dr")
        if len(cont_dl_all) is 0:
            os.remove(img_dl_path+".png")
        else:
            self._save_json_file(img_dl,cont_dl_all,split_name,"_dl")



    def _insert_dict_list_val(self,dict1,key,val):
        if val.__len__() <1:
            pass
        elif dict1 is None or key not in dict1.keys():
            dict1[key] = [val]
        else:
            is_in = False
            check_pt = val[0]
            for lis in dict1[key]:
                cmp_pt = lis[0]
                if int(check_pt[0]) == int(cmp_pt[0]) and int(check_pt[1]) == int(cmp_pt[1]):
                    is_in = True            
            # if not np.array(val ) in np.array( dict1[key],dtype=object):
            if not is_in:
                dict1[key].append(val)

        return  dict1

    def _merge_dict(self,dict1,dict2):
        for key1,val1 in dict1.items():
            if key1 not in dict2.keys():
                dict2[key1] = val1
            else:
                dict2[key1].extend(val1)
        return dict2

    def _split_intersec_anno_part(self, img, intersec_cont, rect, pos):
        part_conts = {}
        for key,conts in intersec_cont.items():  #不同图片处理不同类别，每个类别的标注单独处理
            for cont in conts:
                canvs = np.zeros_like(img, np.uint8)
                cont = np.array(cont,np.int32)
                cv2.drawContours(canvs,[cont],0,(255,255,255),-1)
                map_part_conts = self._split_img_part(canvs, rect, pos)

                for c in map_part_conts[pos]:
                    part_conts = self._insert_dict_list_val(part_conts,key,c)

        return part_conts

    def _split_img_part(self, canv_img, rect, pos):
        r,c = canv_img.shape[:-1]
        img_part = canv_img[  rect[1]:rect[3],rect[0]: rect[2],:]
        conts = []
        map_conts = {pos:conts}
        part_img = cv2.cvtColor(img_part, cv2.COLOR_BGR2GRAY)
        _,part_img_bin = cv2.threshold(part_img, 0, 255, cv2.THRESH_BINARY)
        conts,_ = cv2.findContours(part_img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        #判断conts 删除外接矩形长度小于50
        for cont in conts:
            cont = np.squeeze(cont)
            if len(cont) >2:
                bbox = cv2.minAreaRect(cont)
                max_len =max(bbox[1][0],bbox[1][1])
                if max_len >  self.min_len:
                    map_conts[pos].append(cont)

        return map_conts

    def _split_intersec_anno(self,img,pt,intersec_cont):
        tl_conts = {};tr_conts = {}; dr_conts = {}; dl_conts = {}
        for key,conts in intersec_cont.items():  #不同图片处理不同类别，每个类别的标注单独处理
            for cont in conts:
                canvs = np.zeros_like(img, np.uint8)
                cont = np.array(cont,np.int32)
                cv2.drawContours(canvs,[cont],0,(255,255,255),-1)
                map_part_conts = self._split_img_cont(canvs,pt)

                for c in map_part_conts["_tl"]:
                    tl_conts = self._insert_dict_list_val(tl_conts,key,c)
                for c in map_part_conts["_tr"]:    
                    tr_conts = self._insert_dict_list_val(tr_conts,key,c)
                for c in map_part_conts["_dr"]:
                    dr_conts = self._insert_dict_list_val(dr_conts,key,c)
                for c in map_part_conts["_dl"]:    
                    dl_conts = self._insert_dict_list_val(dl_conts,key,c)

        return tl_conts,tr_conts,dr_conts,dl_conts

    def _split_img_cont(self,canv_img,pt):
        r,c = canv_img.shape[:-1]
        r_cent = pt[1]
        c_cent = pt[0]
        img_tl = canv_img[0:r_cent+self.gap,0:c_cent+self.gap,:]
        img_tr = canv_img[0:r_cent+self.gap,c_cent-self.gap:c,:]
        img_dr = canv_img[r_cent-self.gap:r,c_cent-self.gap:c,:]
        img_dl = canv_img[r_cent-self.gap:r,0:c_cent+self.gap,:]
        map_img = {"_dr":img_dr,"_dl":img_dl,"_tr":img_tr,"_tl":img_tl}
        conts_tl = [];  conts_tr = [];  conts_dr = []; conts_dl = []
        map_conts = {"_dr":conts_dr,"_dl":conts_dl,"_tr":conts_tr,"_tl":conts_tl}

        for pos_name,patch in map_img.items():
            shapes = []
            shape={}
            part_img = cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY)
            _,part_img_bin = cv2.threshold(part_img, 0, 255, cv2.THRESH_BINARY)
            # conts,_ = cv2.findContours(part_img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            conts,_ = cv2.findContours(part_img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
            #判断conts 删除外接矩形长度小于50
            filter_conts = []
            for cont in conts:
                cont = np.squeeze(cont)
                if len(cont) >2:
                    bbox = cv2.minAreaRect(cont)
                    max_len =max(bbox[1][0],bbox[1][1])
                    if max_len >  self.min_len:
                        map_conts[pos_name].append(cont)

        return map_conts


    def _save_json_file(self, patch:np.array, conts,split_name,pos_suffix):
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
        # image_data = self.calc_imageDate(os.path.join(split_root_pathsplit_root_path,split_name+pos_name)+".png")
        image_data = labelme.LabelFile.load_image_file(os.path.join(self.split_root_dir, split_name + pos_suffix) + ".png")
        image_height = patch.shape[0]
        image_width = patch.shape[1]
        file_name = os.path.join(self.split_root_dir, split_name + pos_suffix + ".json")
        lf = labelme.LabelFile()

        lf.save(filename=file_name, shapes=shapes, imagePath=image_path, imageHeight=image_height,
                imageWidth=image_width, imageData=image_data)
        print(f"save json {split_name + pos_suffix}")

    def _get_split_json(self,canv_img,pt,split_name):
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
            part_img_path = os.path.join(self.split_root_dir,split_name+pos_name)+".png"

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
            image_data =  labelme.LabelFile.load_image_file(os.path.join(self.split_root_dir,split_name+pos_name)+".png")
            image_height = patch.shape[0]
            image_width = patch.shape[1]
            file_name = os.path.join(self.split_root_dir,split_name+pos_name+".json")
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


import sys
sys.path.append("D:\\2_MyProjects\\pythonSpt\\SnapBox-CV")
import utils.get_img_file as get_file
import glob
from tqdm import tqdm

if __name__ == "__main__":
    root_path = r"E:\5_Snapbox\8_DLData\L1-L2-front-defect-HH-1\V0.0-标注调整-2\L1-front-HH"
    # save_sub = 'L2-front-HH\L2A_split'
    save_sub = 'L1-front-HH\L1A_split'
    json_path_list =[
    # r"E:\5_Snapbox\8_DLData\L1-L2-front-defect-HH-1\V0.0-标注调整-2\L2-front-HH\测试采集图",
    # r"E:\5_Snapbox\8_DLData\L1-L2-front-defect-HH-1\V0.0-标注调整-2\L2-front-HH\工作采集图",
    # r"E:\5_Snapbox\8_DLData\L1-L2-front-defect-HH-1\V0.0-标注调整-2\L2-front-HH\硬划痕\rmbackground",
    # r"E:\5_Snapbox\8_DLData\L1-L2-front-defect-HH-1\V0.0-标注调整-2\L2-front-HH\工作采集图2",

    r"E:\5_Snapbox\8_DLData\L1-L2-front-defect-HH-1\V0.0-标注调整-2\L1-front-HH\测试采集图",
    r"E:\5_Snapbox\8_DLData\L1-L2-front-defect-HH-1\V0.0-标注调整-2\L1-front-HH\工作采集图",
    r"E:\5_Snapbox\8_DLData\L1-L2-front-defect-HH-1\V0.0-标注调整-2\L1-front-HH\硬划痕\rmbackground",
    r"E:\5_Snapbox\8_DLData\L1-L2-front-defect-HH-1\V0.0-标注调整-2\L1-front-HH\优化添加\L1A_Black",
    r"E:\5_Snapbox\8_DLData\L1-L2-front-defect-HH-1\V0.0-标注调整-2\L1-front-HH\优化添加\L1A_White",

        ]
    scratch_min_len = 25

    labeltxt_path =root_path+"\\label.txt"
    save_path = os.path.join(r'E:\5_Snapbox\8_DLData\L1-L2-front-defect-HH-1\V0.0-标注调整-2', save_sub)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    total =0
    for json_path in tqdm(json_path_list):
        glob_path = os.path.join(json_path,"*.json")
        img_root_path = json_path
        json_files =glob.glob(glob_path)

        # filter 
        js_files = json_files
        for i,j in enumerate((js_files)):
            # if not "196358" in j:
                # continue
            total = total+1
            print(f"{i}/{js_files.__len__()}  split {j} start")
            _, f_name = os.path.split(j)
            anno_split = AnnoSplit(img_root_path,j,save_path,labeltxt_path,gap =64,min_len= scratch_min_len)
            img = anno_split.show_annno()
            y = img.shape[0]//2
            x = img.shape[1]//2

            print(f"split point x={x},y ={y}")
            anno_split.create_parts_gap((x,y))
            print(f"img splited")
            # break

    print(f"split total is {total}")


