from cv2 import Canny, contourArea
import numpy as np
import cv2
import math

img = cv2.imread('CVtask.jpg')

def rescale(frames,scale = 0.5):
     width = int(frames.shape[1]*scale)
     height = int(frames.shape[0]*scale)
     dim = (width,height)
     return cv2.resize(frames,dim,interpolation = cv2.INTER_AREA)
def ModifiedWay(rotateImage, angle):
    
    # Taking image height and width
    imgHeight, imgWidth = rotateImage.shape[0], rotateImage.shape[1]
  
    # Computing the centre x,y coordinates
    # of an image
    centreY, centreX = imgHeight//2, imgWidth//2
  
    # Computing 2D rotation Matrix to rotate an image
    rotationMatrix = cv2.getRotationMatrix2D((centreY, centreX), angle, 1.0)
  
    # Now will take out sin and cos values from rotationMatrix
    # Also used numpy absolute function to make positive value
    cosofRotationMatrix = np.abs(rotationMatrix[0][0])
    sinofRotationMatrix = np.abs(rotationMatrix[0][1])
  
    # Now will compute new height & width of
    # an image so that we can use it in
    # warpAffine function to prevent cropping of image sides
    newImageHeight = int((imgHeight * sinofRotationMatrix) +
                         (imgWidth * cosofRotationMatrix))
    newImageWidth = int((imgHeight * cosofRotationMatrix) +
                        (imgWidth * sinofRotationMatrix))
  
    # After computing the new height & width of an image
    # we also need to update the values of rotation matrix
    rotationMatrix[0][2] += (newImageWidth/2) - centreX
    rotationMatrix[1][2] += (newImageHeight/2) - centreY
  
    # Now, we will perform actual image rotation
    rotatingimage = cv2.warpAffine(
        rotateImage, rotationMatrix, (newImageWidth, newImageHeight))
  
    return rotatingimage

class properties():
  
  
  def coun(self,image):
    list = []
    self.image = image
    imgGrey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thrash = cv2.threshold(imgGrey, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)
    
        x = approx.ravel()[0]
        y = approx.ravel()[1] - 5

        if len(approx) == 4:
        
            x1 ,y1, w, h = cv2.boundingRect(approx)
            aspectRatio = float(w)/h
           # print(aspectRatio)
            if aspectRatio >= 0.95 and aspectRatio <= 1.05:
                contours, _ = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                
                # cv2.drawContours(image, [approx], 0, (0, 0,255), 1)
               
                # cv2.putText(image, "square", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0),2)
                n = approx.ravel() 
                i = 0
            


                for j in n : 
                  if(i % 2 == 0): 
                    x = n[i] 
                    y = n[i + 1] 

               # String containing the co-ordinates. 
                  string = str(x)+ " " +str(y)
                  area = cv2.contourArea(contour)
                  if area > 21000 and area < 300000 :
                  
                   list.append([int(x),int(y)])
                #used if need to see co_ordinates in the image
                #    if(i == 0): 
                #    # text on topmost co-ordinate. 
                #      cv2.putText(image, "", (x, y), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0,255)) 
                #    else: 
                #    # text on remaining co-ordinates. 
                #      cv2.putText(image, string, (x, y),cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0,255)) 
                   i = i +1
    print(f'co_ordiunates of the squares : {list[ : :2]}')              
    return list[ : :2]
  def angle(self,lis):
   pi = 22/7 
   p = abs((lis[2][1]-lis[3][1])/(lis[2][0]-lis[3][0]))
   radians = math.atan(p)
   slope = radians*(180/pi)
   print(f"slope of square :{slope}")
   return slope
  def bound(self,list):
   
   curcumscribed_area = (list[0][1]-list[2][1])*(list[0][1]-list[2][1])
   print(f"circumscribed area of square :{curcumscribed_area}")
   return curcumscribed_area

class need:
 def crop(self,p,png):
  pts = np.array(p)

# (1) Crop the bounding rect
  rect = cv2.boundingRect(pts)
  x,y,w,h = rect
  croped = png[y:y+h, x:x+w].copy()

## (2) make mask
  pts = pts - pts.min(axis=0)

  mask = np.zeros(croped.shape[:2], np.uint8)
  cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

## (3) do bit-op
  dst = cv2.bitwise_and(croped, croped, mask=mask)

## (4) add the white background
  bg = np.ones_like(croped, np.uint8)*255
  cv2.bitwise_not(bg,bg, mask=mask)
  dst2 = bg+ dst

  return dst2

#croping patches  --------------------------------------------------------------------------------------------
nimg = rescale(img)
opu = need()

cropu = opu.crop([[100,428],[100,502],[182,502],[182,428]],nimg)
opt = need()

cropt = opt.crop([[100,500],[100,574],[223,574],[223,500]],nimg)

opd = need()

cropd = opd.crop([[492,260],[492,353],[588,353],[588,260]],nimg)
opx = need()

cropx = opx.crop([[643,569],[643,400],[700,400],[700,569]],nimg)
opy = need()

cropy = opy.crop([[51,330],[51,232],[111,232],[111,330]],nimg)
#-------------------------------------------------------------------------------------------------------------------------------------


om = properties()
ol = om.coun(nimg)
cv2.imshow("rescaled img",nimg)
ol3 = ol[ :4]
ol4 = ol[4:8]
ol2 = ol[8 :12]
ol1 = ol[12:16]
print ("black square prop")
print(ol3)
oa3 = om.angle(ol3)
oar3 = om.bound(ol3)
print("----------------------------")
print ("pink_peach square prop")
print(ol4)
oa4 = om.angle(ol4)
oar4 = om.bound(ol4)
print("----------------------------")

print ("orange square prop")
print(ol2)
oa2 = om.angle(ol2)
oar2 = om.bound(ol2)
print("----------------------------")
print ("green square prop")
print(ol1)
oa1 = om.angle(ol1)
oar1 = om.bound(ol1)
print("----------------------------")

print ("marker_3 properties")
img3= cv2.imread('Ha.jpg')
m3 = properties()
l3 = m3.coun(img3)
a3 = m3.angle(l3)
ar3 = m3.bound(l3)
print("--------------------------------------------------------------------")

img2= cv2.imread('XD.jpg')
print ("marker_2 properties")

m2 = properties()
l2 = m2.coun(img2)
a2 = m2.angle(l2)
ar2 = m2.bound(l2)
print("--------------------------------------------------------------------")

img1= cv2.imread('LMAO.jpg')
print ("marker_1 properties")

m1 = properties()
l1 = m1.coun(img1)
a1 = m1.angle(l1)
ar1 = m1.bound(l1)
print("--------------------------------------------------------------------")

img4= cv2.imread('HaHa.jpg')
print ("marker_4 properties")

m4 = properties()
l4 = m4.coun(img4)
a4 = m4.angle(l4)
ar4 = m4.bound(l4)
print("--------------------------------------------------------------------")

#drawing contours and giving co_ordinates to rescales image---------------------------------------------
nimg1 = rescale(img)
imgGrey = cv2.cvtColor(nimg1, cv2.COLOR_BGR2GRAY)
_, thrash = cv2.threshold(imgGrey, 240, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)
    
        x = approx.ravel()[0]
        y = approx.ravel()[1] - 5

        if len(approx) == 4:
        
            x1 ,y1, w, h = cv2.boundingRect(approx)
            aspectRatio = float(w)/h
           # print(aspectRatio)
            if aspectRatio >= 0.95 and aspectRatio <= 1.05:
                contours, _ = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
              #  print(len(contours))
                cv2.drawContours(nimg1, [approx], 0, (0, 0,255), 1)
               
                cv2.putText(nimg1, "square", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0),2)
                n = approx.ravel() 
cv2.imshow("squares detected",nimg1)
# scaling factor could be used if countors are more accurate
def scale (list1,list2):
    scf = (list1[0][1]-list1[2][1])/(list2[0][1]-list2[2][1])
    return scf

#---------------------------------------------------------------------------------------------------------------------
#overlaping Ha
rot3= ModifiedWay(img3,-(a3+oa3))
r3 = properties()
print("-----------------------------------------------------------------\n rotated 3")
lr3 = r3.coun(rot3)

arr3 = r3.bound(lr3)

op3 = need()

crop3 = op3.crop([[100,506],[505,675],[675,270],[270,100]],rot3)


nimg3 = rescale(crop3,0.344347826)
nimg[233:431,644:842] = nimg3
nimg[399:569,643:701] = cropx

#-------------------------------------------------------------------------------------------------------------------------------

#overlapping XD
rot2= ModifiedWay(img2,(a2+oa2))

r2 = properties()
print("-----------------------------------------------------------------\n rotated 2")
lr2 = r2.coun(rot2)

arr2 = r2.bound(lr2)

crop2 = rot2[123:588,123:588]

nimg2 = rescale(crop2,0.382795699)
nimg[37:215,583:761] = nimg2
  
#-----------------------------------------------------------------------------------------------------------
#overlaping of LMAO

rot1= ModifiedWay(img1,(a1-oa1))
r1 = properties()
print("-----------------------------------------------------------------\n rotated 1")
lr1 = r1.coun(rot1)

arr1 = r1.bound(lr1)

op1 = need()

crop1 = op1.crop(lr1,rot1)


nimg1 = rescale(crop1,scale(ol1,lr1))
nimg[16:266,51:301] = nimg1
nimg[231:330,51:112] = cropy
opz = need()
cropz = opz.crop([[160,200],[160,269],[264,269],[264,200]],nimg)
#---------------------------------------------------------------------------------------------------------------------


#overlaping of HaHa

rot4= ModifiedWay(img4,(a4-oa4))
r4 = properties()
print("-----------------------------------------------------------------\n rotated 4")
lr4 = r4.coun(rot4)

arr4 = r4.bound(lr4)
op4a = need()
crop4a = op4a.crop([[60,272],[271,1000],[1000,44],[445,61]],rot4)


op4 = need()
crop4 = op4.crop([[0,228],[211,613],[596,402],[385,17]],crop4a)


nimg4 = rescale(crop4,0.605704698)
nimg[208:569,165:526] = nimg4
nimg[200:270,160:265] = cropz
nimg[260:354,492:589] = cropd
nimg[500:575,100:224] = cropt
nimg[428:503,100:183] = cropu


final = rescale(nimg,2)

cv2.imshow("overlap",nimg)

cv2.imshow("final",final)
cv2.imwrite("final_output_image.png", final)


cv2.waitKey(0)