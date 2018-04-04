import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as LA


# fig = plt.figure(1) # in inches!
def sobel(img):
    
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
    return sobelx, sobely

# gradient for grayscale image
def gradientGray(img,px):
    Ix , Iy = sobel(img)
    angle = np.arctan2( Iy[px] , Ix[px] )
    edgeStrength = np.absolute(Ix[px]) + np.absolute(Iy[px])
    # edgeStrength = np.sqrt(Ix*Ix + Iy*Iy)
    # px = np.where((edgeStrength > 30))
    return angle , edgeStrength

#gradient of color by u = (rx + gx + bx, ry + gy + by) , m = |rx| + |gx| + |bx| + |ry| + |gy| + |by| 
def gradientUM(img,px): 
    b,g,r = cv2.split(img)

    rx , ry =sobel(r)
    gx , gy =sobel(g)
    bx , by =sobel(b)

    u = rx[px] + gx[px] + bx[px]
    v = ry[px] + gy[px] + by[px]

    angle = np.arctan2( v , u )
    m = (np.absolute(rx[px]) + np.absolute(gx[px]) + np.absolute(bx[px]) + np.absolute(ry[px]) + np.absolute(gy[px]) + np.absolute(by[px]) )/2
    # px = np.where((m > 30))

    sign = np.sign(angle)

    return angle , m , sign

#gradient of color by eigenvectors
def gradientEigen(img, sign , px):

    b,g,r = cv2.split(img)
    rx , ry =sobel(r)
    gx , gy =sobel(g)
    bx , by =sobel(b)

    a = rx[px]*rx[px] 
    a = a + gx[px]*gx[px] 
    a = a + bx[px]*bx[px]
    b = rx[px]*ry[px] 
    b = b + gx[px]*gy[px]
    b = b + bx[px]*by[px]
    c = ry[px]*ry[px] 
    c = c + gy[px]*gy[px]
    c = c + by[px]*by[px]

    trace = a + c
    edgeStrength = np.sqrt(trace)

    lambda1= 0.5 * (trace + np.sqrt((trace*trace)-4*(b*b - a*c)))
    # lambda2= 0.5 * (trace - np.sqrt((trace*trace)-4*(b*b - a*c)))
    # X = np.zeros((img.shape[0] , img.shape[1]))
    X = -b / (a - lambda1)
    X[np.isnan(X)] = 1

    # e = np.ones((2,img.shape[0],img.shape[1])) 
    # ex = X / np.sqrt(X*X + 1)
    # ey = 1 / np.sqrt(X*X + 1)

    # angle = np.arctan2( ey , ex )*sign  #angle in radians
    angle = np.arctan2( 1, X ) *sign #angle in radians

    # px= np.where((lambda2 < 1) & (lambda1 == a))  #horizontal or vertical edge
    # angle[px] = np.arctan(1)
    # px= np.where((lambda2 < 1) & (lambda1 != a))  #other edge
    # angle[px] = np.arctan2( -b[px] , (a[px] - lambda1[px] )  )#angle in radians
    return angle , edgeStrength



def radToDegrees(angle):
    negPos = np.where(angle < 0)
    angle[negPos] = 2* np.pi + angle[negPos]

    d = np.array(np.degrees(angle/10) , dtype='uint16')
    return d


def histogram_intersection(h1,h2):

    sum_min=0.0
    sum_max=0.0
    for i in xrange(0,36):
        hxi_1=h1[i]
        hxi_2=h2[i]

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

        hxi_1=h1[i]
        hxi_2=h2[i]

        if hxi_2 + hxi_1 >0:
            e=( hxi_1 - hxi_2 ) * ( hxi_1 - hxi_2 ) / ( hxi_1 + hxi_2 )
            sum_x_chi_sq = sum_x_chi_sq + e

    return sum_x_chi_sq


def histSimilarity(histseq, s):

    hi_arr = np.zeros((n,n), np.float)
    chisq_arr = np.zeros((n,n), np.float)

    # similarity for all combination of images in the sequence
    for x in xrange(0,n):
        for y in xrange(0,x+1):
            hi_arr[x][y] = histogram_intersection(histseq[x],histseq[y])
            hi_arr[y][x] =  hi_arr[x][y]
            chisq_arr[x][y] =  chi_squared_measure(histseq[x],histseq[y])
            chisq_arr[y][x] =   chisq_arr[x][y]
            print 'Calculating'+s+' histogram similarity for images: ',x,' ',y
            print '\tHis Int = ',hi_arr[x][y]
            print '\tChi Sq = ',chisq_arr[x][y]


    hi_min = hi_arr.min()
    hi_max = hi_arr.max()

    graph = (hi_arr-hi_min)*255.0/hi_max
    g=np.array( graph,dtype=np.uint8 )

    chisq_min = chisq_arr.min()
    chisq_max = chisq_arr.max()

    graph2 = (chisq_arr-chisq_min)*255.0/chisq_max
    g2 = np.array( graph2,dtype=np.uint8 )


    # fig1 = plt.figure(1)
    fig1=plt.figure(1,figsize=(12,6))
    p1=fig1.add_subplot(121)
    plt.xlabel('Images')
    plt.ylabel('Images')
    imgplot1 = plt.imshow(g, interpolation='none')
    imgplot1.set_cmap('rainbow_r')
    p1.set_title('Histogram Intersection similarity')

    p2=fig1.add_subplot(122)
    plt.xlabel('Images')
    plt.ylabel('Images')
    imgplot2 = plt.imshow(g2, interpolation='none') 
    imgplot2.set_cmap('rainbow')
    p2.set_title('Chi-squared Measure similarity')
    
    plt.savefig('resultsH3/Histogram_similarity_graph'+s+'.png')

    plt.suptitle(s, fontsize=12)
    print 'All histograms saved to ./resultsH3/'
    print 'Histogram similarity graphs saved to ./resultsH3/'

    plt.show()
    plt.gcf().clear()


def plot(angle, edgeStrength, px):
    V= np.cos(angle) * edgeStrength
    U= np.sin(angle) * edgeStrength
    plt.quiver(px[1],px[0], U, V, angle, units='x',  pivot='mid', angles='xy', scale_units='xy', scale=8, width=0.5)

def plotAM(angle, edgeStrength, px, dim, s):
    
    fig1=plt.figure(1,figsize=(12,6))
    p1=fig1.add_subplot(121)
    img = np.zeros((dim[0],dim[1]))
    eMax= np.amax(edgeStrength)
    img[px]=255*edgeStrength/eMax
    img = cv2.flip(img , 0)
    imgplot = p1.imshow(img, interpolation='none') 
    imgplot.set_cmap('spectral')
    # p1.imshow()
    p1.set_title('Magnitude')

    # cbar = p1.colorbar(imgplot, ticks=[0, 50, 100, 150, 200, 255], orientation='horizontal')
    # cbar.p1.set_xticklabels(['Edge strength: 0.0, Angle: 0', '0.25, 90 degrees', '0.50, 180 degrees', '0.75, 270 degrees', '1.0, 360 degrees'])  # horizontal colorbar


    p2=fig1.add_subplot(122)
    img2 = np.zeros((dim[0],dim[1]))
    img2[px]=255*(radToDegrees(angle))/360
    img2 = cv2.flip(img2 , 0)

    # i=0
    # for x in px:
    #     for y in x:
    #         if(img[px[0][i]][px[1][i]] !=0):
    #             print img[px[0][i]][px[1][i]] , img2[px[0][i]][px[1][i]]
    #             i=i+1
    #     break

    imgplot = p2.imshow(img2, interpolation='none')
    imgplot.set_cmap('spectral')
    p2.set_title('Angle')
    # fig1.tight_layout()
    plt.suptitle('Grayscale Edge Heatmap', fontsize=12)
    plt.savefig(s)

    plt.show()
    plt.gcf().clear()
    


histseqG=[]
histseqE=[]
histseqUM=[]

# fig = plt.figure(figsize=(9, 6)) # in inches!

n=99

t1=25
t2=85

for i in xrange(0,n):
    ii=str(i+1)
    if i<9:
        path='images/ST2MainHall400'+ii+'.jpg'
    else:
        path='images/ST2MainHall40'+ii+'.jpg'
    t=cv2.imread(path)
    im=cv2.imread(path)
    # im=cv2.resize(im,(640,480))
    im = cv2.flip( im, 0 )

    blurred = cv2.GaussianBlur(im , (9,9),0)
    canny = cv2.Canny(blurred, t1,t2)
    
    # px = edge pixels
    px= np.where(canny!=0)

    print "***********************************"
    print "Loading image: ",path , (i+1)
    print "Calculating gradients:"

############################
    # # GRAY 
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    print "\tGrayscale" 
    angleG, edgeStrengthG = gradientGray(gray, px)
    dG = (radToDegrees(angleG)).ravel()
    histG, bins = np.histogram(dG, bins=36)

############################
    # COLOR HIST : u vector, m magnitude
    print "\tColor: u vector, m magnitude"
    angle, m, sign = gradientUM(blurred, px)
    dUM = (radToDegrees(angle)).ravel()
    histUM, bins = np.histogram(dUM, bins=36)

#############################
    # COLOR HIST : eigenvector
    print "\tColor: Eigenvector"
    angleE, edgeStrengthE = gradientEigen(blurred, sign, px)
    dE = (radToDegrees(angleE)).ravel()
    histE, bins = np.histogram(dE, bins=36)
    
    # displaying quiver gradient plots only for last image in sequence
    if i==n-1:


        plt.suptitle('Color: u,m quiver')
        plot(angle, m, px)
        plt.savefig('resultsH3/STMainHall40'+ii+'_quiverPlot_c_um.png')
        # plt.show()
        plt.gcf().clear()

        plt.suptitle('Color: Eigenvector quiver')
        plot(angleE, edgeStrengthE, px)
        plt.savefig('resultsH3/STMainHall40'+ii+'_quiverPlot_c_eig.png')
        plt.show()
        plt.gcf().clear()

        plt.suptitle('Grayscale quiver')
        plot(angleG, edgeStrengthG, px)
        plt.savefig('resultsH3/STMainHall40'+ii+'_quiverPlot_gray.png')
        # plt.show()
        plt.gcf().clear()

        plotAM(angleG, edgeStrengthG, px, im.shape, 'resultsH3/STMainHall40'+ii+'_magHeatMap_gray.png')
        

        # plt.suptitle('Color-Eigenvector Edge Heatmap: Magnitude, Angle')
        # plotAM(angleE, edgeStrengthE, px, im.shape)
        # plt.savefig('resultsH3/STMainHall40'+ii+'_magHeatMap_c_e.png')
        # plt.show()
        # plt.gcf().clear()


    print "Saving edge histograms"

    plt.suptitle('Histograms for Edge Pixels vs Gradient Angle(36 bin)')
    width = 0.25
    center = (bins[:-1] + bins[1:]) / 2
    
    plt.bar(center, histG, width=width, color = 'grey', label='Gray')
    plt.bar(center+0.25, histUM, align='center', width=width, color = 'turquoise', label='Color (u,m)')
    plt.bar(center+0.5, histE, align='center', width=width, color = 'salmon', label='Color (Eigenvector)')

    histseqG.append(histG)
    histseqE.append(histE)
    histseqUM.append(histUM)

    plt.xlabel('Gradient Angle in degrees/10')
    plt.ylabel('Edges')
    plt.legend()
    plt.savefig('resultsH3/STMainHall40'+ii+'_HistogramPlots.png')
    if i==n-1:
        plt.show()
    plt.gcf().clear()
    print "Saved in ./resultsH3/"
    print ""

histSimilarity(histseqE,"Gray")
histSimilarity(histseqG,"Color_Eigenvector")
histSimilarity(histseqUM,"Color_u-m")