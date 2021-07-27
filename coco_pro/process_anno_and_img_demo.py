"""
缩小原始图片,同时缩小对应的json标签
"""
import glob
import os
import glob
import labelme
import labelme_func
import cv2
import numpy as np
from tqdm import tqdm


json_path_list = [
    r"D:\样本\DLData\L1-frame\V0.0\L1-QKT"
]
save_sub = 'L1-QKT-resize'
save_resize_json_path = os.path.join( r"D:\侧面非划痕缺陷样本\DLData\L1-frame\V0.0", save_sub)

if not os.path.exists(save_resize_json_path):
    os.mkdir(save_resize_json_path)

# 部分限制参数
min_len = 30

total = 0
for root in json_path_list:
    root_json_path = os.path.join(root,"*.json")
    g_paths = glob.glob(root_json_path)

    for p in tqdm(g_paths):
        total = total+1
        data = labelme_func.load_json(p)
        re_img_path = p.replace('json','jpg')
        name = os.path.split(re_img_path)[1][:-4]
        if not os.path.exists(re_img_path):
            print(f"{re_img_path} doesnt exist!!")
            continue
        img = cv2.imdecode(np.fromfile(re_img_path, dtype=np.uint8),cv2.IMREAD_COLOR)
        
        # 处理图片
        half_img = cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0])), cv2.INTER_AREA)

        # 处理标注
        import copy
        shapes =data["shapes"]
        new_shapes = []
        for sh in shapes:
            points =np.array( sh["points"])/np.array([2,1])  # only resize width
            int_pts = points.astype(np.float32)
            bbox = cv2.minAreaRect(int_pts)

            len = max(bbox[1][0], bbox[1][1])
            if len >= min_len  : # and sh["label"] == "HH"
                lp = points.tolist()
                shape = {"label":sh["label"],"points":lp,"group_id":sh["group_id"],"shape_type":sh["shape_type"] ,"flags":sh["flags"]}
                new_shapes.append(copy.deepcopy(shape))
                pass
        length =(new_shapes.__len__())
        if length < 1:
            continue

        #save json
        json_file_name = os.path.join(save_resize_json_path,name  + "_resize.json")
        img_file_path = os.path.join(save_resize_json_path,name  + "_resize.png")

        cv2.imencode('.png', half_img)[1].tofile(img_file_path)
        image_data = labelme.LabelFile.load_image_file(img_file_path)
        h,w = half_img.shape[:2]
        
        lf = labelme.LabelFile()
        lf.save(filename=json_file_name,shapes = new_shapes,imagePath = name+"_resize.png",imageHeight= h,imageWidth=w,imageData= image_data)
        print(f"save {img_file_path}")
        # break

print(f"all resize files {total}")
