import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as LA

histseqG=[]
histseqE=[]
histseqUM=[]
i=0
n=1

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

# gradient for grayscale image
def gradientGray(img):
    Ix , Iy = sobel(img)
    angle = np.arctan2( Iy , Ix )
    edgeStrength = np.absolute(Ix) + np.absolute(Iy)
    # edgeStrength = np.sqrt(Ix*Ix + Iy*Iy)
    px = np.where((edgeStrength > 30))
    return angle , edgeStrength, px

#gradient of color by u = (rx + gx + bx, ry + gy + by) , m = |rx| + |gx| + |bx| + |ry| + |gy| + |by| 
def gradientUM(img): 
    b,g,r = cv2.split(img)

    rx , ry =sobel(r)
    gx , gy =sobel(g)
    bx , by =sobel(b)

    u = rx + gx + bx
    v = ry + gy + by

    angle = np.arctan2( u , v )
    m = (np.absolute(rx) + np.absolute(gx) + np.absolute(bx) + np.absolute(ry) + np.absolute(gy) + np.absolute(by) )/2
    px = np.where((m > 30))
    return angle , m, px

#gradient of color by eigenvectors
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

    trace = a + c
    edgeStrength = np.sqrt(trace)

    lambda1= 0.5 * (trace + np.sqrt((trace*trace)-4*(b*b - a*c)))
    lambda2= 0.5 * (trace - np.sqrt((trace*trace)-4*(b*b - a*c)))



    # eigval = LA.eigvalsh(S[:,:,]) # if eig1 is 0 and eig2 is large,=> step edge. 

    # lambda1 = eigval[:,:,1] #the larger eigen value
    # lambda2 = eigval[:,:,0] 

    X = np.zeros((img.shape[0] , img.shape[1]))

    px= np.where((lambda2 < 1)& (lambda1 == a))  #horizontal or vertical edge
    X[px] = 1
    px= np.where((lambda2 < 1)& (lambda1 != a))  #other edge
    X[px] = -b[px] / (a[px] - lambda1[px])
    # canny=cv2.Canny(img,t1,t2)
    # px=np.where(canny!=0)
    # X[px] = -b[px] / (a[px] - lambda1[px])
    e = np.ones((2,img.shape[0],img.shape[1])) 
    e[0] = X / np.sqrt(X*X + 1)
    e[1] = 1 / np.sqrt(X*X + 1)

    # angle = np.zeros((img.shape[0] , img.shape[1]))    
    # angle = np.arctan2( e[1] , e[0] )  #angle in radians
    angle = np.arctan2( 1, X )  #angle in radians
    # px= np.where((lambda2 < 1) & (lambda1 == a))  #horizontal or vertical edge
    # angle[px] = np.arctan(1)
    # px= np.where((lambda2 < 1) & (lambda1 != a))  #other edge
    # angle[px] = np.arctan2( -b[px] , (a[px] - lambda1[px] )  )#angle in radians

    px=np.where((edgeStrength > 20))

    i=0
    for x in px:
        for y in x:
            print lambda1[px[0][i]][px[1][i]], lambda2[px[0][i]][px[1][i]], angle[px[0][i]][px[1][i]]
            i=i+1
        break
    return angle , edgeStrength, px

def histogramData(angle , px):
    DegAngle = np.zeros((angle.shape[0], angle.shape[1]))
    negPos = np.where(angle < 0)
    DegAngle[negPos] = 2* np.pi + angle[negPos]
    posPos = np.where(angle >= 0)
    DegAngle[posPos] = angle[posPos]
    # Angles in 360 degree format in 36 bin:
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
    graph2 /= graph2.max()/255.0
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


def plot(angle, edgeStrength, px):
    V= np.cos(angle) * edgeStrength
    U= np.sin(angle) * edgeStrength

    # ax.style.use('default')
    plt.quiver(px[1], px[0], U[px], V[px], np.arctan2(V[px], U[px]), angles='uv', pivot='mid')
    # plt.show()
    # Q = plt.quiver(px[1], px[0], U[px], V[px], np.arctan(V, U), angles='xy', units='x')


for i in xrange(0,n):
    path='images/ST2MainHall40'+str(i+20)+'.jpg'
    im=cv2.imread(path)
    im=cv2.resize(im,(640,480))
    im = cv2.flip( im, 0 )

    blurred = cv2.GaussianBlur(im , (9,9),0)

    print "***Loading image: ",path , (i+1)
    print "Saving Quiver plots:"

    plt.style.use('default')
############################
    # GRAY 
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    angleG, edgeStrengthG, pxG = gradientGray(gray)

    plt.suptitle('Grayscale quiver')
    plot(angleG, edgeStrengthG, pxG)
    print "\tGrayscale"
    plt.savefig('resultsH3/STMainHall40'+str(i+20)+'_quiverPlots_gray.png')
    plt.gcf().clear()

    d = histogramData(angleG,pxG)
    histrG = cv2.calcHist([d],[0],None,[36],[0,36])
#############################
    # COLOR HIST : eigenvector
    angleE, edgeStrengthE, pxE = gradientColor(blurred)
    plt.suptitle('Color: Eigenvector quiver')

    plot(angleE, edgeStrengthE, pxE)
    print "\tColor: Eigenvector"
    plt.savefig('resultsH3/STMainHall40'+str(i+20)+'_quiverPlots_c_e.png')
    plt.gcf().clear()

    d = histogramData(angleE,pxE)
    histrE = cv2.calcHist([d],[0],None,[36],[0,36])
############################
    # COLOR HIST : u vector, m magnitude
    u, m, px = gradientUM(blurred)
    plt.suptitle('Color: u,m quiver')
    plot(u, m, px)
    print "\tColor: u vector, m magnitude"
    plt.savefig('resultsH3/STMainHall40'+str(i+20)+'_quiverPlots_c_um.png')
    plt.gcf().clear()

    d = histogramData(u,px)
    histrUM = cv2.calcHist([d],[0],None,[36],[0,36])

# ###########################################################################
    # plt.style.use('seaborn')
    print "Saving edge histograms"

    plt.suptitle('Histograms for Edge Pixels(strength threshold) vs Gradient Angle(36 bin)')
    plt.plot(histrG,color = 'black', label='Gray')
    plt.plot(histrE,color = 'chocolate', label='Color (Eigenvector)')
    plt.plot(histrUM,color = 'coral', label='Color (u,m)')

    histseqG.append(histrG)
    histseqE.append(histrE)
    histseqUM.append(histrUM)

    plt.xlabel('Gradient Angle in degrees/10')
    plt.ylabel('Pixels with high Edge Strength')
    plt.legend()
    plt.savefig('resultsH3/STMainHall40'+str(i+20)+'_HistogramPlots.png')
    plt.gcf().clear()
    print "Saved in ./resultsH3/"
    print ""

    



    # plt.style.use('default')

    # p1 = plt.subplot2grid((4, 4), (0, 0), colspan=2, rowspan=2)
    # p1=fig1.add_subplot(311)

    # # fig1=plt.figure()
    # p2=fig1.add_subplot(211)
    # # p2 = plt.subplot2grid((2, 1), (2, 0), colspan=2, rowspan=2)
    # p2.set_title('Color: EigenVector quiver')
    # plot(angleE, edgeStrengthE, pxE, p2)

    # p3=fig1.add_subplot(212)
    # # p3 = plt.subplot2grid((4, 4), (2, 2), colspan=2, rowspan=2)
    # p3.set_title('Color: u-m quiver')
    # plot(u, m, px, p3)

    # plt.savefig('resultsH3/STMainHall40'+str(i+20)+'_quiverPlots_color.png')
    # # plt.show('resultsH3/STMainHall40'+str(i+20)+'_quiverPlots.png')
    # plt.gcf().clear()


# histSimilarity(histseqE)
# histSimilarity(histseqG)
# histSimilarity(histseqUM)