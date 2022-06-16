import os
import math
import time
import mmcv
import xml.dom.minidom as minidom
from argparse import ArgumentParser
from tqdm import trange

from mmdet.apis import init_detector
from mmdet.apis import inference_detector_huge_image

def makexml(xml_path, model, result, score_thr, img):   
    doc = minidom.Document()
    a = minidom.getDOMImplementation()

    annotation = doc.createElement('annotation') 
    doc.appendChild(annotation)

    source = doc.createElement('source')
    annotation.appendChild(source)
    filename = doc.createElement('filename')
    source.appendChild(filename) 
    message_filename = doc.createTextNode(xml_path.split('/')[-1])
    filename.appendChild(message_filename)
    origin = doc.createElement('origin')
    source.appendChild(origin) 
    message_origin = doc.createTextNode('GF2/GF3')
    origin.appendChild(message_origin)

    research = doc.createElement('research')
    annotation.appendChild(research)
    version = doc.createElement('version')
    research.appendChild(version) 
    message_version = doc.createTextNode('1.0')
    version.appendChild(message_version)
    provider = doc.createElement('provider')
    research.appendChild(provider) 
    message_provider = doc.createTextNode('FAIR1M')
    provider.appendChild(message_provider)
    author = doc.createElement('author')
    research.appendChild(author) 
    message_author = doc.createTextNode('xxffhhzzz')
    author.appendChild(message_author)
    pluginname = doc.createElement('pluginname')
    research.appendChild(pluginname) 
    message_pluginname = doc.createTextNode('FAIR1M')
    pluginname.appendChild(message_pluginname)
    pluginclass = doc.createElement('pluginclass')
    research.appendChild(pluginclass) 
    message_pluginclass = doc.createTextNode('object detection')
    pluginclass.appendChild(message_pluginclass)
    Time = doc.createElement('time')
    research.appendChild(Time) 
    message_time = doc.createTextNode(time.strftime("%Y-%m-%d"))
    Time.appendChild(message_time)

    img = mmcv.imread(img)
    hh, ww = img.shape[:2]

    size = doc.createElement('size')
    annotation.appendChild(size)
    width = doc.createElement('width')
    size.appendChild(width) 
    message_width = doc.createTextNode(str(ww))
    width.appendChild(message_width)
    height = doc.createElement('height')
    size.appendChild(height) 
    message_height = doc.createTextNode(str(hh))
    height.appendChild(message_height)
    depth = doc.createElement('depth')
    size.appendChild(depth) 
    message_depth = doc.createTextNode('3')
    depth.appendChild(message_depth)

    objects = doc.createElement('objects')
    annotation.appendChild(objects)

    for label, result in zip(model.CLASSES, result):
        #res.append((label, result.tolist()))
        res = result.tolist()
        for re in res:
            if re[-1] > score_thr:
                wit = re[2]/2
                len = re[3]/2
                coss = math.cos(-re[4])
                sins = math.sin(-re[4])
                x1 = wit * coss - len * sins + re[0]
                y1 = wit * sins + len * coss + re[1]
                x2 = -wit * coss - len * sins + re[0]
                y2 = -wit * sins + len * coss + re[1]
                x3 = -wit * coss + len * sins + re[0]
                y3 = -wit * sins - len * coss + re[1]
                x4 = wit * coss + len * sins + re[0]
                y4 = wit * sins - len * coss + re[1]
                #print(x1,y1,x2,y2,x3,y3,x4,y4)
                object = doc.createElement('object')
                objects.appendChild(object) 
                coordinate = doc.createElement('coordinate')
                object.appendChild(coordinate) 
                message_coordinate = doc.createTextNode('pixel')
                coordinate.appendChild(message_coordinate)
                type = doc.createElement('type')
                object.appendChild(type) 
                message_type = doc.createTextNode('rectangle')
                type.appendChild(message_type)
                description = doc.createElement('description')
                object.appendChild(description) 
                message_description = doc.createTextNode('None')
                description.appendChild(message_description)
                possibleresult = doc.createElement('possibleresult')
                object.appendChild(possibleresult) 
                name = doc.createElement('name')
                possibleresult.appendChild(name) 
                message_name = doc.createTextNode(label.replace('-',' '))
                name.appendChild(message_name)

                points = doc.createElement('points')
                object.appendChild(points) 

                point1 = doc.createElement('point')
                message_points1 = doc.createTextNode(str(round(x1,6))+','+str(round(y1,6)))
                point1.appendChild(message_points1)
                point2 = doc.createElement('point')
                message_points2 = doc.createTextNode(str(round(x2,6))+','+str(round(y2,6)))
                point2.appendChild(message_points2)
                point3 = doc.createElement('point')
                message_points3 = doc.createTextNode(str(round(x3,6))+','+str(round(y3,6)))
                point3.appendChild(message_points3)
                point4 = doc.createElement('point')
                message_points4 = doc.createTextNode(str(round(x4,6))+','+str(round(y4,6)))
                point4.appendChild(message_points4)
                point5 = doc.createElement('point')
                message_points5 = doc.createTextNode(str(round(x1,6))+','+str(round(y1,6)))
                point5.appendChild(message_points5)

                points.appendChild(point1) 
                points.appendChild(point2)
                points.appendChild(point3)
                points.appendChild(point4)
                points.appendChild(point5)

    f = open(xml_path, 'w')  
    doc.writexml(writer=f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
    f.close()

def calc_result(data):
    if data.find("dota")!=-1:
        f = open("work_dirs/faster_rcnn_orpn_r50_fpn_1x_dota10/dota_result.txt")
        line = f.readline()
        print(line.strip())
        line = f.readline()
        res = line.split("| ")[1:]
        sum = 0
        for r in res:
            sum+=float(r.strip())
        line=line+"| "+str(sum/len(res))[:9]
        print(line)
        f.close()
    elif data.find("fair1m")!=-1:
        f = open("work_dirs/faster_rcnn_orpn_r50_fpn_1x_ms_rr_fair1m/fair1m_result.txt")
        line = f.readline()
        sum = 0
        t = 0
        while line:
            res = line.strip().split()
            sum+=float(res[1])
            if len(res[0])<4 or res[0][0]=='A' or res[0][0]=='B' or res[0][0]=='C':
                print(res[0]+'\t\t'+res[1])
            else:
                print(res[0]+'\t'+res[1])
            line = f.readline()
            t += 1
        print("mAp" + '\t\t' + str(sum/t)[:9])
        f.close()
    else:
        f = open("work_dirs/faster_rcnn_orpn_r50_3x_hrsc/result.txt")
        line = f.readline()
        while line:
            line = f.readline()
            print(line)
        f.close()


def main():
    parser = ArgumentParser()
    parser.add_argument('img_dir', help='Image dir')
    parser.add_argument('output_dir', help='Output dir')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        'split', help='split configs in BboxToolkit/tools/split_configs')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    nms_cfg = dict(type='BT_nms', iou_thr=0.5)
    
    root = args.img_dir
    #calc_result(args.checkpoint)
    for root,_,files in os.walk(root):
            for file,_ in zip(files,trange(len(files))):
                img = os.path.join(root, file)
                result = inference_detector_huge_image(model, img, args.split, nms_cfg)
                makexml(os.path.join(args.output_dir, file[:-3]+'xml'), model, result, args.score_thr, img)  
    print("calc results ...")
    time.sleep(5)
    print("\n")
    calc_result(args.checkpoint)
    

if __name__ == '__main__':

    main()
