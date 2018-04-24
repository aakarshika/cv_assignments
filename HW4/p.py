import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import os
import csv

font = cv2.FONT_HERSHEY_SIMPLEX

display = False

filename=""

# folder="try"
folder="GaitImages"

dfSeq=[]
cntSeq=[]
featuresTable=[]

areaA=[]
perA=[]

areaCA=[]
perCA=[]

defectsAreaA=[]
defectsNA=[]

moments1A=[]
moments1CA=[]
moments2A=[]
moments2CA=[]

def areaTriangle(a, b, c):
    def distance(p1, p2):
        return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

    side_a = distance(a, b)
    side_b = distance(b, c)
    side_c = distance(c, a)
    s = 0.5 * ( side_a + side_b + side_c)
    return math.sqrt(s * (s - side_a) * (s - side_b) * (s - side_c))


def distanceTransformEucl(contour1,img2):
    # distance transform

    px = np.where(img2 >= 0)
    p=np.array(img2.copy(), dtype=np.float32)

    bx2 =   contour1[0][:,0,0]
    by2 =   contour1[0][:,0,1]
    # print len(contour1)
    for x in xrange(1,len(contour1)):
        bx2 =   np.append(bx2, contour1[x][:,0,0])
        by2 =   np.append(by2, contour1[x][:,0,1])

    minp = 256
    maxp = 0
    for i in range(0,len(px[0])):
        x1=px[1][i]
        y1=px[0][i]
        dist = np.array(np.sqrt((bx2-x1)*(bx2-x1)+(by2-y1)*(by2-y1)), dtype=np.float32)
        p[y1,x1] = np.amin(dist)
        if p[y1,x1] >maxp:
            maxp = p[y1,x1] 
        if p[y1,x1] <minp:
            minp = p[y1,x1] 

    dp = (p-minp)*255.0/maxp
    dp = np.array( dp, dtype=np.uint8 )
    cv2.imwrite('results/DistanceTransformEucl/'+filename,dp)
    if display:
        print 'Distance Transform Eucledian'
        cv2.imshow( 'image' , dp )
        cv2.waitKey()
    return p

def distanceTransform2pass(contour1,img1):
    # distance transform

    px1 =   contour1[0][:,0,0]
    py1 =   contour1[0][:,0,1]

    for x in xrange(1,len(contour1)):
        px1 =   np.append(px1, contour1[x][:,0,0])
        py1 =   np.append(py1, contour1[x][:,0,1])

    px = np.where(img1 >=0)
    p1=img1.copy()
    p1[:,:]=255
    p1[[py1,px1]]=0.0

    # 1 2 3
    # 8 0 4
    # 7 6 5

    # 3f 4f 3f
    # 4f 00 4b
    # 3b 4b 3b

    p=p1.copy()

    # forward pass
    for ind in range(0,len(px[0])):
        i=px[0][ind]
        j=px[1][ind]
        if i>=127 or j>=87 or i<=0 or j<=0:
            continue
        D0=p[i][j]
        D1=p[i-1][j-1]
        D2=p[i][j-1]
        D3=p[i+1][j-1]
        D8=p[i-1][j]
        p[i][j] = min(D1+3 ,D2+4 , D3+3 , D8+4 , D0 )
        # p[i][j] = min(D1+1 ,D2+1 , D3+1 , D8+1 , D0 )

    # backward pass
    for ind in range(0,len(px[0])):
        i=px[0][ind]
        j=px[1][ind]
        if i>=127 or j>=87 or i<=0 or j<=0:
            continue
        D0=p[i][j]
        D5=p[i+1][j+1]
        D6=p[i][j+1]
        D7=p[i-1][j+1]
        D4=p[i+1][j]
        p[i][j] = min(D5+3 ,D6+4 , D7+3 , D4+4 , D0 )
        # p[i][j] = min(D5+1 ,D6+1 , D7+1 , D4+1 , D0 )
  
    if display:
        px=np.where(p < 250)
        minp = np.amin(p[px])
        maxp = np.amax(p[px])
        # print minp , maxp
        dp=p.copy()
        dp[px] = (dp[px]-minp)*255.0/maxp
        dp = np.array( dp, dtype=np.uint8)
        print 'Distance Transform 2 pass'
        cv2.imwrite('results/DistanceTransform/'+filename,dp)
        cv2.imshow( 'image' , dp )
        cv2.waitKey()

    return p

def camferMatchScore(cnt1,df2):
    
    bpx = cnt1
    px1 =   bpx[:,0,0]
    py1 =   bpx[:,0,1]
    # print df2[py1,px1]
    p255=np.where(df2 == 255)
    df2[p255] = 1
    
    dm= np.sum(df2[py1,px1])
    return dm


def contour(imgray):
    # contour
    _,contours,hierarchy = cv2.findContours(imgray, 1, cv2.CHAIN_APPROX_NONE)

    c=cv2.drawContours(org.copy(), contours, -1, (0,255,0), 1)
    cv2.imwrite('results/Contours/'+filename,c)
    if display:
        print 'Contours'
        cv2.imshow('image', c)
        cv2.waitKey()
    return contours

def ApproxPolygon(cnt):
    # Poly
    epsilon = 0.01*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    # # k = cv2.isContourConvex(cnt)
    poly = cv2.drawContours(org.copy(),[approx],-1,(255,0,255),2)
    cv2.imwrite('results/ApproxPolygon/'+filename, poly)
    if display:
        print 'Polygon Approximation'
        cv2.imshow('image', poly)
        cv2.waitKey()

def convex(ccnt):
    # convex hull n convexity

    img = org.copy()
    hull = cv2.convexHull(ccnt,returnPoints = False)
    defects = cv2.convexityDefects(ccnt,hull)
    # print defects
    # print "Number of convexity defects:", defects.shape[0]
    defectArea=0
    for j in range(defects.shape[0]):
        s,e,f,d = defects[j,0]
        start = tuple(ccnt[s][0])
        end = tuple(ccnt[e][0])
        far = tuple(ccnt[f][0]) 
        cv2.line(img,start,end,[0,255,0],2)
        cv2.circle(img,far,5,[0,0,255],-1)
        a=areaTriangle(start, end, far)
        defectArea += a
        # print start,end,far,a
        # cv2.putText(im,str(a), far, cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255,0,123))
    # print "Area of defects:", defectArea
    cv2.imwrite('results/Convexity/'+filename,img)
    if display:
        print 'Convex Hull and Deficits of Convexity'
        cv2.imshow('image',img)
        cv2.waitKey()
    return defects.shape[0] , defectArea

def areaPerimeterMoments(cnt):
    # area, permitere
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt,True)
    # print "Area:",area
    # print "Perimeter:",perimeter

    # moments
    M = cv2.moments(cnt)
    # first order
    # print 'First Order:'
    # print 'm10',M['m10']
    # print 'm01',M['m01']
    # # second order
    # print 'Second Order:'
    # print 'm20',M['m20']
    # print 'm02',M['m02']
    # print 'm11',M['m11']
    return (area,perimeter,M)

def curvature(cnt):
    # Curvature

    k=4

    x0=cnt[:,0,0]
    x1=cnt[k:,0,0]
    x1=np.append(x1,cnt[0:k,0,0])
    x2=cnt[m-k:,0,0]
    x2=np.append(x2,cnt[0:m-k,0,0])

    y0=cnt[:,0,1]
    y1=cnt[k:,0,1]
    y1=np.append(y1,cnt[0:k,0,1])
    y2=cnt[m-k:,0,1]
    y2=np.append(y2,cnt[0:m-k,0,1])

    a0=x0
    a2=( a0 + x2 - 2*x1 )/2 #by solving simultaneous equations
    a1=( 4*x1 - 3*a0 - x2 )/2

    b0=y0
    b2=( b0 + y2 - 2*y1 )/2 #by solving simultaneous equations
    b1=( 4*y1 - 3*b0 - y2 )/2

    Ktan = 2*(a1*b2 - b1*a2) / np.power( (a1*a1 + b1*b1), 1.5) 

    mink = np.amin(Ktan)
    maxk = np.amax(Ktan)
    # normKtan = (Ktan - mink)*255.0/maxk
    normKtan = 255 - (np.absolute(Ktan))*255.0/maxk
    normKtan = np.array( normKtan, dtype=np.uint8 )

    imk=imgray.copy()
    imk[:]=255
    imk[y0,x0]=normKtan

    # imgplot2 = plt.imshow(imk, interpolation='none') 
    # imgplot2.set_cmap('RdBu')
    # plt.suptitle('Curvature: High = Red, Low = Blue', fontsize=8)
    # plt.savefig('results/Curvature/'+filename)

    imkc = cv2.applyColorMap(imk, cv2.COLORMAP_HOT)
    cv2.imwrite('results/Curvature/'+filename,imkc)
    if display:
        print 'Curvature: (High = Hotter Red)'
        cv2.imshow('image',imkc)
        cv2.waitKey()
        # plt.show()
    # plt.gcf().clear()

    return normKtan 


def localMaxima(cnt,normKtan):
    # local maxima
    x0=cnt[:,0,0]
    y0=cnt[:,0,1]

    a=normKtan
    k=1
    a1_=normKtan[k:]
    a1_=np.append(a1_,normKtan[0:k])
    a_1=normKtan[m-(k):]
    a_1=np.append(a_1,normKtan[0:m-(k)])

    k=2
    a2_=normKtan[k:]
    a2_=np.append(a2_,normKtan[0:k])
    a_2=normKtan[m-(k):]
    a_2=np.append(a_2,normKtan[0:m-(k)])


    k=2
    a3_=normKtan[k:]
    a3_=np.append(a3_,normKtan[0:k])
    a_3=normKtan[m-(k):]
    a_3=np.append(a_3,normKtan[0:m-(k)])


    localMaxPx = np.where((a > a1_) & (a > a_1) & (a > a2_) & (a > a_2)& (a > a3_) & (a > a_3))
    imLMax = cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB)
    for j in localMaxPx[0]:
        imLMax = cv2.circle(imLMax,(x0[j],y0[j]),1,(255,0,0),-1)
    cv2.imwrite('results/LocalMaxima/'+filename,imLMax)
    if display:
        print 'Local Maxima'
        cv2.imshow('image',imLMax)
        cv2.waitKey()
    return localMaxPx


def plotAlFeatures():
    
    plt.plot(areaA,  'm', label='Number of defects')
    plt.plot(perCA,  'b', label='Total Area of defects')
    plt.legend(loc='best')


    plt.suptitle('Deficits of Convex Hull')
    plt.savefig('results/ConvexHullDefects.png')
    plt.show()
    plt.gcf().clear()


    plt.plot(areaA,  'b', label='Area Silhouettes')
    plt.plot(areaCA, 'b--', label='Area Convex Hull')
    plt.plot(perA,  'r', label='Perimeter Silhouettes')
    plt.plot(perCA,  'r--', label='Perimter Convex Hull')
    plt.legend(loc='best')


    plt.suptitle('Area and Perimeter')
    plt.savefig('results/AreaPerimeter.png')
    plt.show()
    plt.gcf().clear()

    plt.plot(moments1A[:,0],  'm', label='M(1,0) Silhouettes')
    plt.plot(moments1A[:,1],  'm', label='M(0,1) Silhouettes')
    plt.plot(moments1CA[:,0], 'm--', label='M(1,0) Convex Hull')
    plt.plot(moments1CA[:,1], 'm--', label='M(0,1) Convex Hull')
    plt.legend(loc='best')


    plt.suptitle('First Order Moments')
    plt.savefig('results/FirstOrderMoments.png')
    plt.show()
    plt.gcf().clear()


    plt.plot(moments2A[:,0],  'y', label='M(2,0) Silhouettes')
    plt.plot(moments2A[:,1],  'y', label='M(0,2) Silhouettes')
    plt.plot(moments2A[:,2],  'g', label='M(1,1) Silhouettes')
    plt.plot(moments2CA[:,0], 'y--', label='M(2,0) Convex Hull')
    plt.plot(moments2CA[:,1], 'y--', label='M(0,2) Convex Hull')
    plt.plot(moments2CA[:,2], 'g--', label='M(1,1) Convex Hull')
    plt.legend(loc='best')


    plt.suptitle('Second Order Moments')
    plt.savefig('results/SecondOrderMoments.png')
    plt.show()
    plt.gcf().clear()




print "Computing area, permiter, moments, contours, approx-polygon, convex hull, curvature, local maxima, and distance transform for:"

for fname in sorted(os.listdir(folder)):
    filename = fname    
    im = cv2.imread(os.path.join(folder,filename))
    imgray = cv2.imread(os.path.join(folder,filename) , 0)
    print filename
    
    features={}

    if filename == "00000172.png":
        cv2.namedWindow('image',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 600,600)
        display = True

    features['filename']=filename

    if im is not None:
        org = im.copy()

        ret,thresh = cv2.threshold(imgray,127,255,cv2.THRESH_BINARY)

        contours=contour(thresh)

        cL=len(contours)
        cnt=contours[cL-1]
        # print cnt
        cntSeq.append(cnt)
        m=len(cnt)

        ApproxPolygon(cnt)

        # IMAGE FEATURES
        defectsN, defectsA = convex(cnt)
        features['defectsN']= defectsN
        features['defectsArea']= defectsA
        defectsNA.append(defectsN)
        defectsAreaA.append(defectsA)
        a,p,M=areaPerimeterMoments(cnt)

        features['imgArea']= a
        features['imgPerimeter']= p
        features['m10']=M['m10']
        features['m01']=M['m01']
        features['m20']=M['m20']
        features['m02']=M['m02']
        features['m11']=M['m11']
        areaA.append(a)
        perA.append(p)
        moments1A.append([M['m10'],M['m01']])
        moments2A.append([M['m20'],M['m02'],M['m11']])

        # CONVEX HULL FEATURES
        cHull = cv2.convexHull(cnt)
        a,p,M=areaPerimeterMoments(cHull)

        features['CHArea']= a
        features['CHPerimeter']= p
        features['cm10']=M['m10']
        features['cm01']=M['m01']
        features['cm20']=M['m20']
        features['cm02']=M['m02']
        features['cm11']=M['m11']
        areaCA.append(a)
        perCA.append(p)
        moments1CA.append([M['m10'],M['m01']])
        moments2CA.append([M['m20'],M['m02'],M['m11']])

        featuresTable.append(features)

        normKtan = curvature(cnt)
        localMaxPx = localMaxima(cnt,normKtan)

        df=distanceTransform2pass(contours,thresh)
        # df=distanceTransformEucl(contours,thresh)
        dfSeq.append(df)

        if display:
            distanceTransformEucl(contours,thresh)

cv2.destroyAllWindows()


moments1A = np.array(moments1A)
moments1CA = np.array(moments1CA)

moments2A = np.array(moments2A)
moments2CA = np.array(moments2CA)
plotAlFeatures()

n=len(dfSeq)

keys = featuresTable[0].keys()
with open('results/features.csv', 'wb') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(featuresTable)
print "\nFeatures saved in features.csv file in 'results' directory."



score = np.zeros((n,n), dtype=np.float)
for y in xrange(0,n):
    for x in xrange(0,n):
        score[x][y]=camferMatchScore(cntSeq[x],dfSeq[y])

hmax = score.max()
# print hmin, hmax
score = (score)*255.0/hmax
g=np.array( score,dtype=np.uint8 )

plt.xlabel('Images')
plt.ylabel('Images')
imgplot2 = plt.imshow(g, interpolation='none') 
imgplot2.set_cmap('rainbow_r')
plt.suptitle('Chamfer Matching for all contour boundaries with Distance Transform', fontsize=8)
plt.savefig('results/ChamferMatching_similarity_graph.png')

print "Chamfer Matching Confusion Matrix saved to 'results'"

plt.show()
plt.gcf().clear()




