import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as LA

histseq=[]
i=0
n=10

t1=35
t2=90
rt1=30
rt2=80
gt1=30
gt2=80
bt1=30
bt2=80

def sobel(img):
    
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
    return sobelx, sobely

def plot(angle, edgeStrength, px):
    V= np.cos(angle) * edgeStrength
    U= np.sin(angle) * edgeStrength
    # print '*******'
    # i=0
    # for x in px:
    #     for y in x:
    #         print U[px[0][i]][px[1][i]] , V[px[0][i]][px[1][i]] , edgeStrength[px[0][i]][px[1][i]], angle[px[0][i]][px[1][i]]
    #         i=i+1
    #     break
    # for x in angle:
    #     print x
    plt.figure()
    # Q = plt.quiver(np.arange(640), np.arange(480), U, V, np.arctan2(V, U), angles='uv', units='x')
    plt.quiver(px[1], px[0], U[px], V[px], np.arctan2(V[px], U[px]), angles='uv', pivot='mid')
    # Q = plt.quiver(px[1], px[0], U[px], V[px], np.arctan(V, U), angles='xy', units='x')
    plt.show()

def gradientGray(img):
    Ix , Iy = sobel(img)
    angle = np.arctan2( Iy , Ix )
    edgeStrength = np.sqrt(Ix*Ix + Iy*Iy)
    px = np.where((edgeStrength > 30))
    return angle , edgeStrength, px


def save(fname , img):
    cv2.imwrite('results/'+fname ,anglePic)


def gradientColor(img):
    b,g,r = cv2.split(img)

    rx , ry =sobel(r)
    gx , gy =sobel(g)
    bx , by =sobel(b)

    S = np.zeros(( img.shape[0], img.shape[1],2,2))
    S[:,:,0,0] = rx*rx + gx*gx + bx*bx
    S[:,:,0,1] = rx*ry + gx*gy + bx*by
    S[:,:,1,0] = S[:,:,0,1]
    S[:,:,1,1] = ry*ry + gy*gy + by*by

    a=S[:,:,0,0]
    b=S[:,:,0,1]
    c=S[:,:,1,1]

    # trace = a + c
    edgeStrength = np.sqrt(a + c)

    eigval = LA.eigvalsh(S[:,:,]) # if eig1 is 0 and eig2 is large,=> step edge. 

    lambda1 = eigval[:,:,1] #the larger eigen value
    lambda2 = eigval[:,:,0] 

    X = np.zeros((img.shape[0] , img.shape[1]))

    px= np.where((lambda2 < 1) & (lambda1 == a))  #horizontal or vertical edge
    X[px] = 1
    px= np.where((lambda2 < 1) & (lambda1 != a))  #other edge
    X[px] = -b[px] / (a[px] - lambda1[px])

    # angle = np.arctan2( X[:] ,np.ones(X.shape) )

    e = np.ones((2,img.shape[0],img.shape[1])) 
    e[0] = X / np.sqrt(X*X + 1)
    e[1] = 1 / np.sqrt(X*X + 1)

    angle = np.arctan2( e[1] , e[0] )  #angle in radians
    # angle = np.arctan2( lambda2 , lambda1) * 180 / np.pi


    px=np.where((edgeStrength > 20) & (X !=0) )

    # i=0
    # for x in px:
    #     for y in x:
    #         print e[0][px[0][i]][px[1][i]] , e[1][px[0][i]][px[1][i]], angle[px[0][i]][px[1][i]]
    #         i=i+1
    #     break
    

    return angle , edgeStrength, px


for i in xrange(0,1):
    path='images/ST2MainHall40'+str(i+20)+'.jpg'
    im=cv2.imread(path)
    im=cv2.resize(im,(640,480))
    im = cv2.flip( im, -1 )

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blurredgray = cv2.GaussianBlur(gray, (9,9),0)
    angle, edgeStrength, px = gradientGray(blurredgray)
    plot(angle, edgeStrength, px)

    blurred = cv2.GaussianBlur(im , (9,9),0)
    angle, edgeStrength, px = gradientColor(blurred)
    plot(angle, edgeStrength, px)

# anglePic = normalise (edgeStrength , 255)
# save('EM'+str(i+20)+'a95.png' , anglePic)
