import cv2
import numpy as np
from matplotlib import pyplot as plt

imgpath= "images/myimg.png"

import wx
app = wx.App()
frame = wx.Frame(None, -1, 'win')
frame.SetDimensions(0,0,200,50)
# Create text input
dlg = wx.TextEntryDialog(frame, 'Image Path','Enter image name')
dlg.SetValue("images/myimg.png")
if dlg.ShowModal() == wx.ID_OK:
    imgpath= dlg.GetValue()
dlg.Destroy()


font = cv2.FONT_HERSHEY_SIMPLEX

t=cv2.imread(imgpath)
img=cv2.resize(t,(1280, 960))
cv2.putText(img,'1.1.3. Move mouse over image. Press "q" to continue. ',(30,30), font, 1,(0,0,255),2,cv2.LINE_AA)

original = img.copy()




# hist_blue, bins = np.histogram(img[:][:][0], bins=256)
# hist_green, _ = np.histogram(img[:][:][1], bins=256)
# hist_red, _ = np.histogram(img[:][:][2], bins=256)

# width = 0.25
# center = (bins[:-1] + bins[1:]) / 2
# plt.bar(center, hist_blue.ravel(), width=width, color = 'blue', label='Blue')
# plt.bar(center+width, hist_green.ravel(), width=width, color = 'green', label='Green')
# plt.bar(center+(width*2), hist_red.ravel(), width=width, color = 'red', label='Red')

histr = cv2.calcHist([img],[0],None,[256],[0,256])
plt.plot(histr,color = 'b')
plt.xlim([0,256])
histr = cv2.calcHist([img],[1],None,[256],[0,256])
plt.plot(histr,color = 'g')
plt.xlim([0,256])
histr = cv2.calcHist([img],[2],None,[256],[0,256])
plt.plot(histr,color = 'r')
plt.xlim([0,256])
plt.suptitle('1.1.2 RGB histograms for image. Press "q" to continue.')
plt.savefig('results/RGBHistogram.png')
plt.show()
plt.gcf().clear()

img_size=img.shape

#Mouse callback function
def draw_box(event,x,y,flags,param):
    if event == cv2.EVENT_MOUSEMOVE:
        # print x, y
        copy= original.copy()
        cv2.rectangle(copy,(x-6,y-6),(x+6,y+6),(0,0,0),1)
        cv2.circle(copy,(x,y),0,(200))
        cv2.putText(copy,'P=('+str(x)+' '+str(y)+')',(x-15,y-35), font, 0.5,(0,0,0),1,cv2.LINE_AA)
        r=img.item(y,x,2)
        g=img.item(y,x,1)
        b=img.item(y,x,0)
        cv2.putText(copy,'RGB=['+str(r)+' '+str(g)+' '+str(b)+']',(x-15,y-15), font, 0.5,(0,0,0),1,cv2.LINE_AA)
        i=  (r+b+g)/3
        cv2.putText(copy,'I='+str(i),(x-15,y+25), font, 0.5,(0,0,0),1,cv2.LINE_AA)
        window=img[y-5:y+5,x-5:x+5]
        # print window
        m=np.mean(window)
        sd=np.std(window)
        cv2.putText(copy,'Mean='+str(round(m,1))+' SD='+str(round(sd,1)),(x-15,y+45), font, 0.5,(0,0,0),1,cv2.LINE_AA)
        cv2.imshow('image',copy )


cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_box)
cv2.imshow('image',img)
while(1):
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break

cv2.destroyAllWindows()



histseq=[]
i=0
n=99
fig2=plt.figure(1)
print '2.A. Loading images and calculating histograms...'

for i in range (0,n):
    if i<9:
        path='images/ST2MainHall400'+str(i+1)+'.jpg'
    else:
        path='images/ST2MainHall40'+str(i+1)+'.jpg'
    t=cv2.imread(path)
    im=cv2.resize(t,(640,480))
    # imgseq.append(im)
    print 'Saving histogram for image: ', path
    intensity = np.zeros([480,640], dtype=np.uint16)



    # split function does not work here because the desired array is uint16 and split returns uint8
    r=np.array(im[:,:,2] ,dtype=np.uint16 )
    g=np.array(im[:,:,1] ,dtype=np.uint16 )
    b=np.array(im[:,:,0] ,dtype=np.uint16 )

    # Change from previously submitted Homework
    intensity =    ((r >> 5) << 6 ) + ( ( g >> 5 ) << 3) + ( b >> 5) 

    intensity_ravel =intensity.ravel()

    histr , bins = np.histogram( intensity_ravel, 512)

    fig2.add_subplot(1,2,1)
    plt.hist(intensity_ravel,bins=512,color = 'black')

    #########################################

    plt.xlabel('Intensity')
    plt.ylabel('Pixels')
    plt.xlim([0,512])

    fig2.add_subplot(1,2,2)
    plt.imshow(im,interpolation='none')
    plt.savefig('results/STMainHall40'+str(i+1)+'_histogram.png')
    plt.gcf().clear()
    # print histr
    histseq.append(histr)

def histogram_intersection(h1,h2):

    sum_min=0.0
    sum_max=0.0
    for i in xrange(0,512):
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
    for i in xrange(0,512):

        hxi_1=h1[i]
        hxi_2=h2[i]

        if hxi_2 + hxi_1 >0:
            e=( hxi_1 - hxi_2 ) * ( hxi_1 - hxi_2 ) / ( hxi_1 + hxi_2 )
            sum_x_chi_sq = sum_x_chi_sq + e

    return sum_x_chi_sq



# cv2.destroyAllWindows()

fig = plt.figure(1)

hi_arr = np.zeros((n,n), np.float)
chisq_arr = np.zeros((n,n), np.float)

for x in xrange(-1,n):
    for y in xrange(0,x+1):
        hi_arr[x][y] = histogram_intersection(histseq[x],histseq[y])
        hi_arr[y][x] =  hi_arr[x][y]
        chisq_arr[x][y] =  chi_squared_measure(histseq[x],histseq[y])
        chisq_arr[y][x] =   chisq_arr[x][y]
        print '2.B. Calculating histogram similarity for images: ',str(x+1),' ',str(y+1)
        print '\tHis Int = ',hi_arr[x][y]
        print '\tChi Sq = ',chisq_arr[x][y]





# graph = hi_arr*255.0
# g=np.array( graph,dtype=np.uint8 )

# graph2 = chisq_arr
# graph2 /= graph2.max()/256
# g2 = np.array( graph2,dtype=np.uint8 )


hi_min = hi_arr.min()
hi_max = hi_arr.max()

graph = (hi_arr-hi_min)*255.0/hi_max
g=np.array( graph,dtype=np.uint8 )

chisq_min = chisq_arr.min()
chisq_max = chisq_arr.max()

graph2 = (chisq_arr-chisq_min)*255.0/chisq_max
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

plt.suptitle('2.C. Blue= High similarity. Red= Low similarity.', fontsize=16)
print 'All histograms saved to ./results/'
print 'Histogram similarity graph saved to ./results/'

plt.show()

