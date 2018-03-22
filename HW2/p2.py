import cv2
import numpy as np
from matplotlib import pyplot as plt
# imgseq=[]
histseq=[]
i=0
n=50
fig2=plt.figure(1)
print 'Loading images and calculating histograms...'

for i in range (0,n):
	path='images/ST2MainHall40'+str(i+20)+'.jpg'
	t=cv2.imread(path)
	im=cv2.resize(t,(640, 480))
	# imgseq.append(im)
	print 'Saving histogram for image: ', path
	intensity = np.zeros([480,640], dtype=np.uint16)
	r=np.array(im[:,:,2] ,dtype=np.uint16 )
	g=np.array(im[:,:,1] ,dtype=np.uint16 )
	b=np.array(im[:,:,0] ,dtype=np.uint16 )

	intensity =    r*2 + g/4 + b/32 

	histr = cv2.calcHist([intensity],[0],None,[512],[0,512])
	fig2.add_subplot(1,2,1)
	plt.plot(histr,color = 'black')
	plt.xlabel('Intensity')
	plt.ylabel('Pixels')
	plt.xlim([0,512])

	fig2.add_subplot(1,2,2)
	plt.imshow(im,interpolation='none')
	plt.savefig('results/STMainHall40'+str(i+20)+'_histogram.png')
	plt.gcf().clear()
	histseq.append(histr)

def histogram_intersection(h1,h2):

	sum_min=0.0
	sum_max=0.0
	for i in xrange(0,512):
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
	for i in xrange(0,512):

		hxi_1=h1[i][0]
		hxi_2=h2[i][0]

		if hxi_2 + hxi_1 >0:
			e=( hxi_1 - hxi_2 ) * ( hxi_1 - hxi_2 ) / ( hxi_1 + hxi_2 )
			sum_x_chi_sq = sum_x_chi_sq + e

	return sum_x_chi_sq



# cv2.destroyAllWindows()

fig = plt.figure(1)

hi_arr = np.zeros((n,n), np.float)
chisq_arr = np.zeros((n,n), np.float)

for x in xrange(0,n):
	for y in xrange(0,x+1):
		hi_arr[x][y] = histogram_intersection(histseq[x],histseq[y])
		hi_arr[y][x] = 	hi_arr[x][y]
		chisq_arr[x][y] =  chi_squared_measure(histseq[x],histseq[y])
		chisq_arr[y][x] = 	chisq_arr[x][y]
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

