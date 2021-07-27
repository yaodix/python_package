
import json
import numpy as np
import cv2
from pathlib import Path
import base64
import os
import labelme

class CocoProc:
    """coco格式标注文件的处理
    """
    def get_all_label_cls(self, json_path_list):
        """获取所有标签列表，按字母顺序排列

        Args:
            json_path_list: 待计算的json文件列表

        Return:
            所有有效（shape不为空）标签列表
        """
        cls_set = set()
        for json_path in json_path_list:
            # print(f"{json_path}")
            data, code = self.load_json(json_path)
            shapes = data["shapes"]
            if shapes is None:
                continue
            for label_obj in shapes:
                label = label_obj["label"]
                cls_set.add(label)

        cls_list = np.array(list(cls_set))
        cls_list.sort()
        return cls_list

    def get_all_label_cls_cnt(self, json_path_list ):
        """获取所有标签和其数量
        """
        label_cnt_dict = {}
        for json_path in json_path_list:
            data, code = self.load_json(json_path)
            shapes = data["shapes"]
            if shapes is None:
                continue
            for label_obj in shapes:
                label = label_obj["label"]
                if label in label_cnt_dict.keys():
                    label_cnt_dict[label] = label_cnt_dict[label]+1
                else:
                    label_cnt_dict[label] = 1
                        
        return label_cnt_dict

    def draw_coco_anno(self, img, json_path, label_cls_filter, color : tuple, draw_type, thickness = 1):
        """绘制coco标签

        Args:
            img: 待绘制标签的图像
            json_path:
            label_cls_filter: 指定绘制的标签列表
            color:
            draw_type: must one of "fill","contour"

        Return:
            绘制标签的图像
        """
        with open(json_path, "r") as f:
            data = json.load(f)
            shapes = data["shapes"]
            drawed_img = []
            if shapes is None:
                return drawed_img

            for label_obj in shapes:
                label = label_obj["label"]
                if label in label_cls_filter:
                    poly_pts = label_obj["points"]
                    arr = np.array(poly_pts, np.int32)
                    if draw_type == "fill":
                        drawed_img = cv2.fillPoly(img, [arr], color)
                    elif draw_type == "contour":
                        drawed_img = cv2.polylines(img, [arr], True,color, thickness)

        return  drawed_img

    def set_img_coco_anno(self,src_json_path, target_img_path):
        """将json文件标签设置到指定图像文件

        Args:
            src_json_path : 需要迁移的json标注文件
            target_img_path : 迁移json标注信息的到指定目标图像
        Return:
            None
        """

        with open(src_json_path, "r") as f:
            data = json.load(f)
            if not os.path.exists(target_img_path):
                print(f"{target_img_path} not exist!!!")

            # 替换imageData,imagePath
            img_name = Path(target_img_path).name
            data["imagePath"] = img_name
            data["imageData"] = self._calc_imagedate(target_img_path)
            # 保存json
            new_json_path = Path(target_img_path).with_suffix('.json')
            with open(new_json_path, "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"{new_json_path} saved!")
    
    def _calc_imagedate(self, img_path):
        """计算图像base64编码：部分版本labelimg标注文件中有imageData项
        """
        data = labelme.LabelFile.load_image_file(img_path)
        image_data = base64.b64encode(data).decode('utf-8')
        return image_data

    def load_json(self, json_file_path, verbose = False):
        """加载json文件

        如果文件不是utf-8编码，则使用默认解码方式    
        Arags:
            verbose: 是否显示文件非utf-8编码的提示信息
        Return:
            data: 
            code: 编码方式 "utf-8" 或 "no utf-8" 
        """
        code ="utf-8"
        try:
            data = json.load(open(json_file_path, "r", encoding='utf-8'))
        except Exception as ex:
            if verbose:
                print("no utf-8 decode ")
            data = json.load(open(json_file_path, "r"))
            code = "no utf-8"
        return data, code

    def gen_label_txt(self, cls_array:np.array, save_file_name:str):
        # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
        with open(save_file_name,'w') as f: 
            f.write( "__ignore__\n")  # 末尾”\n“表示换行
            for i, cls in enumerate(cls_array):
                if i == len(cls_array)-1:
                    f.write(cls)    #末行去除空格
                else:
                    f.write(cls+"\n")    #末尾”\n“表示换行

    def transfer_to_utf8(self, json_file_path):
        """将json文件编码转为utf-8编码
        """
        data ,code = self.load_json(json_file_path,True)
        jd = json.dumps(data,ensure_ascii=False)
        f2 = open(json_file_path,"w",encoding='utf-8')
        f2.write(jd)
        f2.close()                
        
    def get_cls_per_file(self, json_file_path):
        """获取单张图片的标签类别
        """
        data ,code = self.load_json(json_file_path)
        cls_set = set()
        shapes = data["shapes"]
        if shapes is None:
            return 
        for label_obj in shapes:
            label = label_obj["label"]
            cls_set.add(label)

        cls_list = np.array(list(cls_set))
        return cls_list

    
from tqdm import tqdm
import shutil
if __name__ == "__main__":
    import glob
    import os

    json_dir = r"D:\侧面非划痕缺陷样本\DLData\LFPFL-frame\V0.1\L1-V0.1\L1-LF"
    _save_dir =  r"D:\侧面非划痕缺陷样本\DLData\LFPFL-frame\V0.1\L1-V0.1\otherLabel"

    if not os.path.exists(_save_dir):
        os.mkdir(_save_dir)
    # if not os.path.exists(LF_save_dir):
    #     os.mkdir(LF_save_dir)
        pass
        
    coco_pro = COCOProc()
    json_files = glob.glob(os.path.join(json_dir,"*.json"))
    for j in tqdm(json_files):
        _, json_name = os.path.split(j)
        img_name = json_name.replace("json","jpg")
        img_src_path = os.path.join(json_dir,img_name)

        labels = coco_pro.get_cls_per_file(j)
        # img = cv2.imdecode(np.fromfile(img_src_path,np.uint8),cv2.IMREAD_COLOR)

        if  not  "PFL" in labels and not "LF" in labels and not "KLF" in labels and not "DQ" in labels:
            json_save_path = os.path.join(_save_dir,json_name)
            img_save_path = os.path.join(_save_dir,img_name)
            shutil.move(j, json_save_path)
            shutil.move(img_src_path, img_save_path)
                # pass
            print(f" move in  { j }")     
        else:
            # json_save_path = os.path.join(LF_save_dir,json_name)
            # img_save_path = os.path.join(LF_save_dir,img_name)

            # shutil.move(j, json_save_path)
            # shutil.move(img_src_path, img_save_path)
            # print(f" move in  { j }")     
            pass
        # break
        
        
        # break


