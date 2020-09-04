import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt

def Question_4_a(img_in, thick, border_colour, font_colour):

    gray = img_in


    contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    img2= np.ones((img_in.shape[0],img_in.shape[1],3), np.uint8)


    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            if gray[i,j]<255:
                img2[i,j,0]=font_colour[2]
                img2[i,j,1]=font_colour[1]
                img2[i,j,2]=font_colour[0]
            else:
                img2[i, j, 0] = 255
                img2[i, j, 1] = 255
                img2[i, j, 2] = 255


    for i in range(1,len(contours)):
        img2= cv2.drawContours(img2,contours[i],-1,(border_colour[2],border_colour[1],border_colour[0]),thick)


    return cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)





def Question_4_b(img_in,shadow_size,shadow_magnitude,orientation):
    degrees = orientation

    shadSize=shadow_size
    sigma=shadow_magnitude
    slope = np.tan(np.radians(degrees))

    if degrees==0 or degrees==360:
        xDir=1*shadSize
        yDir=0
    elif degrees>0 and degrees<90: #tan pos #xDir and yDir should be pos #we want a ratio so we can scale to shadow size
        xDir = shadSize/(1**2+slope**2)*1
        yDir = shadSize/(1**2+slope**2)*slope #ratio of new hypotenuse over current hypotenuse multiply x and y directions
        #xDir = 1
        #yDir=slope

    elif degrees==90:
        xDir=0
        yDir=1*shadSize
    elif degrees>90 and degrees<180:#tan neg


        xDir = shadSize / (1 ** 2 + slope ** 2) * -1
        yDir = shadSize / (1 ** 2 + slope ** 2) * -slope

    elif degrees == 180:
        xDir=-1*shadSize
        yDir=0
    elif degrees >180 and degrees < 270: #tan pos
        xDir = shadSize / (1 ** 2 + slope ** 2) * -1
        yDir = shadSize / (1 ** 2 + slope ** 2) * -slope
    elif degrees == 270:
        yDir=-1*shadSize
        xDir=0
    elif degrees>270 and degrees<360: #tan neg
        xDir = shadSize / (1 ** 2 + slope ** 2) * -1
        yDir = shadSize / (1 ** 2 + slope ** 2) * slope
    img = img_in
    img2 = cv2.GaussianBlur(img,(9,9),sigma)

    tm = np.float32([[1,0,xDir],[0,1,yDir]])
    img2=cv2.warpAffine(img2,tm,(img2.shape[1],img2.shape[0]))



    result=np.zeros((img.shape[0],img.shape[1]),np.uint8)
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if img[i,j]<255:
                result[i,j]=255
            else:
                result[i,j]=img2[i,j]

    return result





def Question_5(scene_img, pokemon_string, location, width):



    (centerX,centerY) = location
    bg=scene_img
    bg = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)
    pokemon=pokemon_string



    width=width





    if pokemon == "Flygon":
        img = cv2.imread("Flygon.JPG", 1)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        (y, x, c) = img.shape

        thresh=200
        thresh2=120
        mask = np.zeros((y, x))


        #black is 0

        for i in range(y):
            for j in range(x):
                #green less than a certain amount turn it white
                if img[i, j, 1]<thresh:
                    mask[i, j] = 1

                if img[i, j, 0] > thresh2 and img[i, j, 2] > thresh2:
                    mask[i,j]=1
        kernel = np.ones((10,10),np.uint8)
        mask = cv2.erode(mask, kernel, iterations = 1)
        kernel = np.ones((15,15),np.uint8)
        mask = cv2.dilate(mask, kernel, iterations = 1)
        kernel = np.ones((5,5),np.uint8)
        mask = cv2.dilate(mask, kernel, iterations = 1)



        for i in range(y-int(y*.25),y):
            for j in range(x):
                if mask[i,j] == 1:
                    mask[i,j]=0

        result = np.zeros((y,x,4), np.uint8)

        for i in range(y):
            for j in range(x):
                if mask[i,j] == 1:
                    result[i,j,1]=img[i,j,1]
                    result[i, j, 0] = img[i, j, 0]
                    result[i, j, 2] = img[i, j, 2]
                    result[i,j,3] = 1
                else:
                    result[i,j,3] = 0



    elif pokemon == "Pikachu":
        img = cv2.imread("Pikachu.jpg", -1)

        (y, x, c) = img.shape

        thresh = 200
        thresh2 = 120
        mask = np.zeros((y, x))


        # black is 0

        for i in range(y):
            for j in range(x):

                if img[i, j, 2] > 185:
                    mask[i, j] = 1
                if img[i, j, 0] < 80 and img[i, j, 1] < 80 and img[i, j, 2] < 80:
                    mask[i, j] = 1

        kernel = np.ones((5
                          , 5))

        mask = cv2.dilate(mask, kernel, iterations=1)

        result = np.zeros((y, x, 4), np.uint8)



        for i in range(y):
            for j in range(x):
                if mask[i, j] == 1:

                    result[i, j, 1] = img[i, j, 1]
                    result[i, j, 0] = img[i, j, 0]
                    result[i, j, 2] = img[i, j, 2]
                    result[i,j,3] = 1
                else:
                    result[i,j,3] = 0



    elif pokemon == "Muk":
        img = cv2.imread("Muk.jpg")

        (y, x, c) = img.shape

        thresh = 200
        thresh2 = 120

        mask = np.ones((y, x))
        for i in range(int(y * .25)):
            for j in range(x):
                mask[i, j] = 0

        for i in range(y):
            for j in range(x):
                if img[i, j, 2] < 200 and img[i, j, 1] > 150 and img[i, j, 0] < 120:
                    mask[i, j] = 0
                if img[i, j, 0] < 50:
                    mask[i, j] = 1

        kernel = np.ones((5, 5))
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.erode(mask, kernel, iterations=2)

        result = np.zeros((y, x, 4), np.uint8)
        for i in range(y):
            for j in range(x):
                if mask[i, j] == 1:
                    result[i, j, 1] = img[i, j, 1]
                    result[i, j, 0] = img[i, j, 0]
                    result[i, j, 2] = img[i, j, 2]
                    result[i,j,3] = mask[i,j]
    elif pokemon == "Jigglypuff":
        img = cv2.imread("jigglypuff.jpg")

        (y, x, c) = img.shape

        thresh = 200
        thresh2 = 120
        mask = np.zeros((y, x))

        for i in range(int(y * .4), y):
            for j in range(x):

                if img[i, j, 0] > 90 and img[i, j, 1] > 90 and img[i, j, 2] > 165:
                    mask[i, j] = 1
                if img[i, j, 0] < 80 and img[i, j, 1] < 80 and img[i, j, 2] < 80:
                    mask[i, j] = 1
                if img[i, j, 2] < 50:
                    mask[i, j] = 1
                if img[i, j, 0] > 80 and img[i, j, 0] < 160 and img[i, j, 1] > 80 and img[i, j, 1] < 160 and img[
                    i, j, 2] > 80 and img[i, j, 2] < 160:
                    mask[i, j] = 1
                if img[i, j, 0] > 120 and img[i, j, 1] > 120:
                    mask[i, j] = 1

        kernel = np.ones((5, 5))
        mask = cv2.erode(mask, kernel, iterations=1)
        kernel = np.ones((8, 8))
        mask = cv2.dilate(mask, kernel, iterations=1)

        result = np.zeros((y, x, 4), np.uint8)
        for i in range(y):
            for j in range(x):
                result[i,j,3]=mask[i,j]
                if mask[i, j] == 1:
                    result[i, j, 1] = img[i, j, 1]
                    result[i, j, 0] = img[i, j, 0]
                    result[i, j, 2] = img[i, j, 2]
    else:
        print("Invalid pokemon name")
        exit(0)






    #cv2.imshow("w", img)
    #cv2.waitKey()
    scale=width/y
    nH =   img.shape[0]*scale
    nW =   width
    result = cv2.resize(result,(int(nW), int(nH)))


    #top corner of where pokemon image should start
    tempX = centerX-(int(result.shape[1]/2))
    tempY = centerY-(int(result.shape[0]/2))


    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if result[i,j,3] !=0:
                bg[tempY+i,tempX+j,0]=result[i,j,0]
                bg[tempY+i,tempX+j,1]=result[i,j,1]
                bg[tempY+i,tempX+j,2]=result[i,j,2]

    bg = cv2.cvtColor(bg,cv2.COLOR_BGR2RGB)

    return bg





#bg=cv2.imread("wormhole.jpg")

#img_out=Question_5(bg,"Pikachu",(int(bg.shape[1]/2),int(bg.shape[0]/2)),400)
#cv2.imshow("w", img_out)
#plt.imshow(img_out)
#plt.show()