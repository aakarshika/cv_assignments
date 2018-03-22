import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as LA

histseqG=[]
histseqE=[]
histseqUM=[]
i=0
n=50

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

    # Q = plt.quiver(np.arange(640), np.arange(480), U, V, np.arctan2(V, U), angles='uv', units='x')
    plt.quiver(px[1], px[0], U[px], V[px], np.arctan2(V[px], U[px]), angles='uv', pivot='mid')
    # Q = plt.quiver(px[1], px[0], U[px], V[px], np.arctan(V, U), angles='xy', units='x')

def gradientGray(img):
    Ix , Iy = sobel(img)
    angle = np.arctan2( Iy , Ix )
    edgeStrength = np.absolute(Ix) + np.absolute(Iy)
    # edgeStrength = np.sqrt(Ix*Ix + Iy*Iy)
    px = np.where((edgeStrength > 30))
    return angle , edgeStrength, px

def gradientUM(img):
    b,g,r = cv2.split(img)

    rx , ry =sobel(r)
    gx , gy =sobel(g)
    bx , by =sobel(b)

    u = np.zeros((2,img.shape[0],img.shape[1]))
    u[0] = rx + gx + bx
    u[1] = ry + gy + by

    angle = np.arctan2( u[1] , u[0] )
    m = (np.absolute(rx) + np.absolute(gx) + np.absolute(bx) + np.absolute(ry) + np.absolute(gy) + np.absolute(by) )/2
    px = np.where((m > 30))
    return angle , m, px


def gradientColor(img):

    # img = cv2.flip( img, 1 )
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

    # X = np.zeros((img.shape[0] , img.shape[1]))
    angle = np.zeros((img.shape[0] , img.shape[1]))

    # px= np.where((lambda2 < 1) & (lambda1 == a))  #horizontal or vertical edge
    # X[px] = 1
    # px= np.where((lambda2 < 1) & (lambda1 != a))  #other edge
    # X[px] = -b[px] / (a[px] - lambda1[px])

    # angle = np.arctan2( X[:] ,np.ones(X.shape) )

    # e = np.ones((2,img.shape[0],img.shape[1])) 
    # e[0] = X / np.sqrt(X*X + 1)
    # e[1] = 1 / np.sqrt(X*X + 1)

    # e = np.ones((2,img.shape[0],img.shape[1])) 
    # e[0] = lambda1 / np.sqrt(lambda1*lambda1 + lambda2*lambda2)
    # e[1] = lambda2 / np.sqrt(lambda1*lambda1 + lambda2*lambda2)
    
    # angle = np.arctan2( e[1] , e[0] )  #angle in radians
    px= np.where((lambda2 < 1) & (lambda1 == a))  #horizontal or vertical edge
    angle[px] = 1
    px= np.where((lambda2 < 1) & (lambda1 != a))  #other edge
    angle[px] = np.arctan2( -b[px] , (a[px] - lambda1[px] )  )#angle in radians

    # px=np.where((edgeStrength > 20) & (X !=0) )
    px=np.where((edgeStrength > 20) & (angle !=0) )

    # i=0
    # for x in px:
    #     for y in x:
    #         print e[0][px[0][i]][px[1][i]] , e[1][px[0][i]][px[1][i]], angle[px[0][i]][px[1][i]]
    #         i=i+1
    #     break
    return angle , edgeStrength, px

def histogramData(angle , px):
    DegAngle = np.zeros((angle.shape[0], angle.shape[1]))
    negPos = np.where(angle < 0)
    DegAngle[negPos] = 2* np.pi + angle[negPos]
    posPos = np.where(angle >= 0)
    DegAngle[posPos] = angle[posPos]
    # d = np.zeros((1,px[1].size))
    # d[0] = edgeStrength[px]
    d = np.array(np.degrees(DegAngle[px]/10) , dtype='uint16')
    # i=0
    # for y in px[0]:
    #     print d[0][i], d[1][i]
    #     i=i+1
    return d


def histogram_intersection(h1,h2):

    sum_min=0.0
    sum_max=0.0
    for i in xrange(0,36):
        hxi_1=h1[i][0]
        hxi_2=h2[i][0]

        min_i = min(hxi_1,hxi_2)
        max_i = max(hxi_1,hxi_2)

        sum_min=sum_min + min_i
        sum_max=sum_max + max_i

        # print sum_min,sum_max
    hi=sum_min/sum_max
    # print hi
    return hi

def chi_squared_measure(h1,h2):

    sum_x_chi_sq=0.0
    for i in xrange(0,36):

        hxi_1=h1[i][0]
        hxi_2=h2[i][0]

        if hxi_2 + hxi_1 >0:
            e=( hxi_1 - hxi_2 ) * ( hxi_1 - hxi_2 ) / ( hxi_1 + hxi_2 )
            sum_x_chi_sq = sum_x_chi_sq + e

    return sum_x_chi_sq


def histSimilarity(histseq):

    fig=plt.figure()

    hi_arr = np.zeros((n,n), np.float)
    chisq_arr = np.zeros((n,n), np.float)

    for x in xrange(0,n):
        for y in xrange(0,x+1):
            hi_arr[x][y] = histogram_intersection(histseq[x],histseq[y])
            hi_arr[y][x] =  hi_arr[x][y]
            chisq_arr[x][y] =  chi_squared_measure(histseq[x],histseq[y])
            chisq_arr[y][x] =   chisq_arr[x][y]
            print 'Calculating histogram similarity for images: ',x,' ',y
            print '\tHis Int = ',hi_arr[x][y]
            print '\tChi Sq = ',chisq_arr[x][y]




    graph = hi_arr*255.0
    g=np.array( graph,dtype=np.uint8 )

    graph2 = chisq_arr
    graph2 /= graph2.max()/256
    g2 = np.array( graph2,dtype=np.uint8 )

    p1=fig.add_subplot(121)
    plt.xlabel('Images')
    plt.ylabel('Images')
    imgplot = plt.imshow(g, interpolation='none') 
    imgplot.set_cmap('rainbow_r')
    p1.set_title('Histogram Intersection similarity')

    p2=fig.add_subplot(122)
    plt.xlabel('Images')
    plt.ylabel('Images')
    imgplot = plt.imshow(g2, interpolation='none') 
    imgplot.set_cmap('rainbow')
    p2.set_title('Chi-squared Measure similarity')
    plt.savefig('results/Histogram_similarity_graph.png')

    plt.suptitle('Blue= High similarity. Red= Low similarity.', fontsize=16)
    print 'All histograms saved to ./results/'
    print 'Histogram similarity graph saved to ./results/'

    plt.show()

# fig2=plt.figure(1)
# fig1=plt.figure(2)

for i in xrange(0,n):
    path='images/ST2MainHall40'+str(i+20)+'.jpg'
    im=cv2.imread(path)
    im=cv2.resize(im,(640,480))
    im = cv2.flip( im, -1 )

    print "***"

    # fig2.add_subplot(1,4,1)

    # GRAY 
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blurredG = cv2.GaussianBlur(gray, (9,9),0)
    angleG, edgeStrengthG, pxG = gradientGray(blurredG)
    d = histogramData(angleG,pxG)
    histrG = cv2.calcHist([d],[0],None,[36],[0,36])
    # plt.plot(histrG,color = 'black', label='Gray Edge pixels')

    # COLOR HIST : eigenvector
    blurred = cv2.GaussianBlur(im , (9,9),0)
    angleE, edgeStrengthE, pxE = gradientColor(blurred)
    d = histogramData(angleE,pxE)
    histrE = cv2.calcHist([d],[0],None,[36],[0,36])
    # plt.plot(histrE,color = 'b', label='Color Edge Pixels (Eigenvector)')

    # COLOR HIST : u vector, m magnitude
    u, m, px = gradientUM(blurred)
    d = histogramData(u,px)
    histrUM = cv2.calcHist([d],[0],None,[36],[0,36])
    # plt.plot(histrUM,color = 'r', label='Color Edge Pixels (u , m)')
    
    # plt.legend()

    # fig2.add_subplot(1,4,2)
    plot(angleG, edgeStrengthG, pxG)

    # fig2.add_subplot(1,4,3)
    plot(angleE, edgeStrengthE, pxE)

    # fig2.add_subplot(1,4,4)
    plot(u, m, px)

    # plt.savefig('results/STMainHall40'+str(i+20)+'_histogram.png')
    histseqG.append(histrG)
    histseqE.append(histrE)
    histseqUM.append(histrUM)

    # plt.show()
    # plt.gcf().clear()

# histSimilarity(histseqE)
# histSimilarity(histseqG)
# histSimilarity(histseqUM)