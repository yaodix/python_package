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

class AnnoSplit():
    def __init__(self,img_root_path,json_path,
                 split_root_path,label_path , min_len):
        self.json_path = json_path
        self.img_path =os.path.join(img_root_path, self._load_json()["imagePath"])
        self.img_name =  self._load_json()["imagePath"]
        self.split_root_path = split_root_path
        self.shapes = self._load_json()["shapes"]
        self.map_name_index = self._get_label_map(label_path)
        self.map_index_name = dict(zip(self.map_name_index.values(),self.map_name_index.keys()))
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
    def create_parts(self,pt):
        if not os.path.exists(self.split_root_path) :
            os.makedirs(self.split_root_path)
        split_name = self.img_name[:-4]
        img_tl_path = os.path.join(self.split_root_path,split_name+"_tl")
        img_tr_path = os.path.join(self.split_root_path,split_name+"_tr")
        img_dr_path = os.path.join(self.split_root_path,split_name+"_dr")
        img_dl_path = os.path.join(self.split_root_path,split_name+"_dl")
        #split img file
        img = cv2.imdecode(np.fromfile(self.img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        r,c = img.shape[:-1]
        y_cent = r_cent = pt[1]
        x_cent = c_cent = pt[0]
        img_tl = img[0:r_cent,0:c_cent,:]
        img_tr = img[0:r_cent,c_cent:c,:]
        img_dr = img[r_cent:r,c_cent:c,:]
        img_dl = img[r_cent:r,0:c_cent,:]
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
            if (cont[:,0] <= x_cent).all() and (cont[:,1] <=y_cent).all():   # tl ,label:conts
                cont = cont-[0,0]      #shift pts
                conts_tl = self._insert_dict_list_val(conts_tl,label,cont)
            elif (cont[:,0] >= x_cent).all() and (cont[:,1] <= y_cent).all():  # tr
                cont = cont-[x_cent,0]
                conts_tr = self._insert_dict_list_val(conts_tr,label,cont)
            elif (cont[:,0]  >= x_cent).all() and (cont[:,1] >= y_cent).all(): # dr
                cont = cont-[x_cent,y_cent]
                conts_dr = self._insert_dict_list_val(conts_dr,label,cont)
            elif (cont[:,0]  <= x_cent).all() and (cont[:,1] >= y_cent).all():  # dl ,label:conts
                cont = cont-[0,y_cent]                              #shift pts
                conts_dl = self._insert_dict_list_val(conts_dl,label,cont)
            else:
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
            self._save_json_file(img_tl,cont_tl_all,split_root_path,split_name,"_tl")
        if len(cont_tr_all) is 0:
            os.remove(img_tr_path+".png")
        else:
            self._save_json_file(img_tr,cont_tr_all,split_root_path,split_name,"_tr")

        if len(cont_dr_all) is 0:
            os.remove(img_dr_path+".png")
        else:
            self._save_json_file(img_dr,cont_dr_all,split_root_path,split_name,"_dr")
        if len(cont_dl_all) is 0:
            os.remove(img_dl_path+".png")
        else:
            self._save_json_file(img_dl,cont_dl_all,split_root_path,split_name,"_dl")

        pass
    def _insert_dict_list_val(self,dict1,key,val):
        if val.__len__() <1:
            pass
        elif dict1 is None or key not in dict1.keys():
            dict1[key] = [val]
        else:
            dict1[key].append(val)

        return  dict1

    def _merge_dict(self,dict1,dict2):
        for key1,val1 in dict1.items():
            if key1 not in dict2.keys():
                dict2[key1] = val1
            else:
                dict2[key1].extend(val1)
        return dict2

    def _split_intersec_anno(self,img,pt,intersec_cont):
        tl_conts = {};tr_conts = {}; dr_conts = {}; dl_conts = {}
        for key,conts in intersec_cont.items():  #不同图片处理不同类别
            canvs = np.zeros_like(img, np.uint8)
            for cont in conts:
                cont = np.array(cont,np.int32)
                cv2.drawContours(canvs,[cont],0,(255,255,255),-1)
            map_part_conts = self._split_img_cont(canvs,pt)

            tl_conts = self._insert_dict_list_val(tl_conts,key,map_part_conts["_tl"])
            tr_conts = self._insert_dict_list_val(tr_conts,key,map_part_conts["_tr"])
            dr_conts = self._insert_dict_list_val(dr_conts,key,map_part_conts["_dr"])
            dl_conts = self._insert_dict_list_val(dl_conts,key,map_part_conts["_dl"])
        for k,v in tl_conts.items():
            tl_conts[k] = v[0]
        for k,v in tr_conts.items():
            tr_conts[k] = v[0]
        for k,v in dr_conts.items():
            dr_conts[k] = v[0]
        for k,v in dl_conts.items():
            dl_conts[k] = v[0]
        return tl_conts,tr_conts,dr_conts,dl_conts

    def _split_img_cont(self,canv_img,pt):
        r,c = canv_img.shape[:-1]
        r_cent = pt[1]
        c_cent = pt[0]
        img_tl = canv_img[0:r_cent,0:c_cent,:]
        img_tr = canv_img[0:r_cent,c_cent:c,:]
        img_dr = canv_img[r_cent:r,c_cent:c,:]
        img_dl = canv_img[r_cent:r,0:c_cent,:]
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

def click_split(event,x,y,flags,param):
    img = param[0]
    name = param[1]
    h,w = img.shape[:-1]
    x_cent,y_cent = (int(w/2),int(h/2))
    len = 100
    if event == cv2.EVENT_MOUSEMOVE:
        show_img = copy.deepcopy(img)
        show_img = cv2.line(show_img,(x,0),(x,h),(0,255,0),3)
        show_img = cv2.line(show_img,(0,y),(w,y),(0,255,0),3)
        show_img = cv2.line(show_img,(x_cent,y_cent-len),(x_cent,y_cent+len),(0,255,0),3)
        show_img = cv2.line(show_img,(x_cent-len,y_cent),(x_cent+len,y_cent),(0,255,0),3)
        cv2.putText(show_img,f"{x},{y}",(100,100),1,4,(0,255,0),3)
        cv2.imshow("win", show_img)
    elif event == cv2.EVENT_LBUTTONUP:
        print(f"split point x={x},y ={y}")
        anno_split.create_parts((x,y))
        split_pt_dict[name] = [x,y]
        print(f"img splited")

        pass


import glob
if __name__ == "__main__":
    root_path = r"E:\5_Snapbox\8_DLData\A2-back-defect-HH-1"
    json_path= r"E:\5_Snapbox\8_DLData\A2-back-defect-HH-1\V0.0"
    save_path = r'E:\5_Snapbox\8_DLData\A2-back-defect-HH-1\V0.1'

    glob_path = os.path.join(json_path,"*.json")
    # json_file =os.path.join(root_path,"20191014121529196358-iPhone5S-L2-255-1.0-40-6500-0-A1.json")
    # img_path = os.path.join(root_path,"20191014121529196358-iPhone5S-L2-255-1.0-40-6500-0-A1.jpg")
    img_root_path = json_path
    labeltxt_path =root_path+"\\label.txt"

    split_root_path = save_path+"\\split"
    split_pt_file = os.path.join(save_path,"split.txt")

    split_pt_dict = {}

    if not os.path.exists(split_pt_file):
        f = open(split_pt_file,'w')
        f.close()
    for line in open(split_pt_file):  #load split points data
        l = line.strip()
        name,x,y = l.split(',')
        split_pt_dict[name] =[x,y]

    json_files =glob.glob(glob_path)
    js_files = []
    for js in json_files:
        _,n = os.path.split(js)
        if not "coco" in n:
            js_files.append(js)

    for i,j in enumerate(js_files[:70]): # 69 4562785 area-B-no-4
        print(f"{i}/{js_files.__len__()}  split {j} start")
        _, f_name = os.path.split(j)
        anno_split = AnnoSplit(img_root_path,j,split_root_path,labeltxt_path,min_len= 15)
        img = anno_split.show_annno()
        cv2.namedWindow("win",cv2.WINDOW_NORMAL|cv2.WINDOW_KEEPRATIO)
        params = [img,f_name]
        cv2.setMouseCallback('win', click_split,params)
        cv2.imshow("win",img)

        k =  cv2.waitKey()
        if k == 32: #space next
            continue
        elif k == 27: #esc quit
            with open(split_pt_file, 'w') as f:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
                for k,v in split_pt_dict.items():
                    f.write(f"{k},{v[0]},{v[1]}\n")  # 末尾”\n“表示换行
            break
    with open(split_pt_file, 'w') as f:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
        for k, v in split_pt_dict.items():
            f.write(f"{k},{v[0]},{v[1]}\n")  # 末尾”\n“表示换行


