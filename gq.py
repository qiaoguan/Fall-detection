#Author: qiaoguan(https://github.com/qiaoguan)
import cv2
import sys
sys.path.append('python')
import darknet as dn
import time

def array_to_image(arr):
    arr = arr.transpose(2,0,1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = (arr/255.0).flatten()
    data = dn.c_array(dn.c_float, arr)
    im = dn.IMAGE(w,h,c,data)
    return im

def detect(net, meta, image, thresh=.24, hier_thresh=.5, nms=.45):
    boxes = dn.make_boxes(net)
    probs = dn.make_probs(net)
    num =   dn.num_boxes(net)
    dn.network_detect(net, image, thresh, hier_thresh, nms, boxes, probs)
    res = []
    for j in range(num):
        for i in range(meta.classes):
            if probs[j][i] > 0:
                res.append((meta.names[i], probs[j][i], (boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h)))
    res = sorted(res, key=lambda x: -x[1])
    dn.free_ptrs(dn.cast(probs, dn.POINTER(dn.c_void_p)), num)
    return res

def isFall(w,h):
    if float(w)/h>=1.1:
        return True
    else:
        return False

#open the input video file
input_movie=cv2.VideoCapture('cs4.mp4')

length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

# Create an output movie file (make sure resolution/frame rate matches input video!)
#get fps the size 
fps = input_movie.get(cv2.CAP_PROP_FPS)  
size = (int(input_movie.get(cv2.CAP_PROP_FRAME_WIDTH)),   
        int(input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT))) 

#define the type of the output movie  
output_movie = cv2.VideoWriter('out_cs4.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)  
#output_movie = cv2.VideoWriter('output_cs1.avi', -1, fps, size)

# load network and weights
net = dn.load_net("cfg/yolo.cfg", "yolo.weights", 0)
meta = dn.load_meta("cfg/coco.data")

res=[]
frame_number=0
while True:
    # Grab a single frame of video
    ret, frame = input_movie.read()
    frame_number += 1

    # Quit when the input video file ends
    if not ret:
        break
    '''
    # detect per 2 frame
    if frame_number%2==0:
        continue
    '''
    # append all the coordinate of the detected person to res
    im = array_to_image(frame)
    start=time.time()
    r = detect(net, meta, im)
    print('the whole running time is: '+str(time.time()-start))
    res=[]
    for item in r:
        if item[0]=='person' or item[0]=='dog' or item[0]=='cat' or item[0]=='horse':
            res.append(item)
    # if multiple exist, and there also contains person,  preserve person only!
    print('--------------')
    print(res)
    if len(res)>1:
        for item in res:
            if item[0]=='person':
                res=[]
                res.append(item)
                break
                
    # get the max rectangle
    result=[]
    maxArea=0
    if len(res)>1:
        for item in res:
            if item[2][2]*item[2][3]>maxArea:
                maxArea=item[2][2]*item[2][3]
                result=item
    elif len(res)==1:
        result=res[0]   
    #draw the result 
    if(len(result)>0):      
        # label the result
        left=int(result[2][0]-result[2][2]/2)
        top=int(result[2][1]-result[2][3]/2)
        right=int(result[2][0]+result[2][2]/2)
        bottom=int(result[2][1]+result[2][3]/2)
        
        #whether fall?
        if isFall(result[2][2],result[2][3]):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255))
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, 'Warning!!!', (left + 6, bottom - 6), font, 0.5, (255, 0, 0), 1)
        else:
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
    '''
    # label the result
    for item in res:
        # Draw a box around the face
        name=item[0]
        
        left=int(item[2][0]-item[2][2]/2)
        top=int(item[2][1]-item[2][3]/2)
        right=int(item[2][0]+item[2][2]/2)
        bottom=int(item[2][1]+item[2][3]/2)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255))
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
    '''
    #Display the result
    cv2.imshow('Fall detection',frame)   
    # Write the resulting image to the output video file
    print("Writing frame {} / {}".format(frame_number, length))
    output_movie.write(frame)
    
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# All done!
input_movie.release()
cv2.destroyAllWindows()
