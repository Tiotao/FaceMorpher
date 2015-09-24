import facetracker
import numpy as np
import matplotlib.pyplot as plt
import Image
import argparse
import cv2

INTERMEDIATE_STEPS = 20
TOTAL_FEATURE = 66
TRIANGLES = [[20,21,23],[21,22,23],[0,1,36],[15,16,45],[0,17,36],[16,26,45],[17,18,37],[25,26,44],[17,36,37],[26,44,45],[18,19,38],[24,25,43],[18,37,38],[25,43,44],[19,20,38],[23,24,43],[20,21,39],[22,23,42],[20,38,39],[23,42,43],[21,22,27],[21,27,39],[22,27,42],[27,28,42],[27,28,39],[28,42,47],[28,39,40],[1,36,41],[15,45,46],[1,2,41],[14,15,46],[28,29,40],[28,29,47],[2,40,41],[14,46,47],[2,29,40],[14,29,47],[2,3,29],[13,14,29],[29,30,31],[29,30,35],[3,29,31],[13,29,35],[30,32,33],[30,33,34],[30,31,32],[30,34,35],[3,4,31],[12,13,35],[4,5,48],[11,12,54],[5,6,48],[10,11,54],[6,48,59],[10,54,55],[6,7,59],[9,10,55],[7,58,59],[9,55,56],[8,57,58],[8,56,57],[7,8,58],[8,9,56],[4,31,48],[12,35,54],[31,48,49],[35,53,54],[31,49,50],[35,52,53],[31,32,50],[34,35,52],[32,33,50],[33,34,52],[33,50,51],[33,51,52],[48,49,60],[49,60,50],[50,60,61],[50,51,61],[51,52,61],[61,62,52],[52,53,62],[53,54,62],[54,55,63],[55,56,63],[56,63,64],[56,57,64],[64,65,57],[57,58,65],[58,59,65],[48,59,65],[66,19,18],[66,18,17],[66,17,0],[67,66,0],[67,0,1],[67,1,2],[67,2,3],[67,3,68],[68,3,4],[68,4,5],[68,5,6],[68,6,7],[68,7,69],[69,7,8],[69,8,9],[69,9,70],[70,9,10],[70,10,11],[70,11,12],[70,12,13],[70,13,71],[71,13,14],[71,14,15],[71,15,16],[71,16,72],[72,16,26],[72,26,25],[72,25,24],[73,24,72],[73,23,24],[73,20,23],[73,19,20],[73,19,66], [60,65,61],[61,65,64],[61,64,62],[64,62,63],[36,37,41],[37,41,38],[41,38,40],[38,40,39],[42,43,47],[43,47,44],[44,47,46],[44,46,45],[48,60,65],[62,63,54]]
CORNERS = [(0, 0), (0, 220), (0, 440), (220, 440), (440, 440), (440, 220), (440, 0), (220, 0)]


def readImage(path):
	img = Image.open(path)
	gray = img.convert('L')
	img = np.asanyarray(img)
	gray = np.asarray(gray)
	return img, gray

def getFeaturePoints(tracker1, tracker2, gray1, gray2):
	tracker1.update(gray1)
	feature1 = tracker1.get2DShape()[0]
	tracker2.update(gray2)
	feature2 = tracker2.get2DShape()[0]
	feature_pair = []
	for i in range(TOTAL_FEATURE):
		p1 = (feature1[i].item(0), feature1[i+TOTAL_FEATURE].item(0))
		p2 = (feature2[i].item(0), feature2[i+TOTAL_FEATURE].item(0))
		feature_pair.append([p1, p2])
	return feature_pair

def interpolatePts(feature_pair):
	step = float(1) / float( INTERMEDIATE_STEPS + 1 )
	steps = []
	intermediate_feat = []
	for i in xrange(0, INTERMEDIATE_STEPS + 2):
		steps.append(step * i)
	for r in xrange(0, len(steps)):
		temp_feat = []
		for pair in feature_pair:
			ratio = steps[r]
			new_x = pair[0][0] * (1-ratio) + pair[1][0] * ratio
			new_y = pair[0][1] * (1-ratio) + pair[1][1] * ratio
			temp_feat.append((new_x, new_y))
		
		temp_feat.extend(CORNERS)

		intermediate_feat.append(temp_feat)
	return intermediate_feat

def warpImage(original, feats, tri, img_path):
	image = cv2.imread(img_path)
	white = (255, 255, 255)
	rows,cols,ch = image.shape
	masked_image = np.zeros(image.shape, dtype=np.uint8)
	for t in tri:
		old_a = original[t[0]]
		old_b = original[t[1]]
		old_c = original[t[2]]
		new_a = feats[t[0]]
		new_b = feats[t[1]]
		new_c = feats[t[2]]
		pts1 = np.float32([old_a,old_b,old_c])
		pts2 = np.float32([new_a,new_b,new_c])
		M = cv2.getAffineTransform(pts1,pts2)
		dst = cv2.warpAffine(image,M,(cols,rows))
		# cv2.imshow('masked image', dst)
		mask = np.zeros(image.shape, dtype=np.uint8)
		roi_corners = np.array([[new_a, new_b, new_c]], dtype=np.int32)
		cv2.fillPoly(mask, roi_corners, white)
		masked = cv2.bitwise_and(dst, mask)
		masked_image = cv2.bitwise_or(masked_image, masked)
	# cv2.imshow('masked image', masked_image)
	# cv2.waitKey()
	# cv2.destroyAllWindows()
	return masked_image

	
def combineImages(features, tri, image_path1, image_path2):
	start_feat = features[0]
	end_feat = features[INTERMEDIATE_STEPS + 1]
	step = float(1) / float( INTERMEDIATE_STEPS + 1 )
	steps = []
	frames = []
	for i in xrange(0, INTERMEDIATE_STEPS + 2):
		ratio = step * i
		current_feat = features[i]
		startImg = warpImage(start_feat, current_feat, tri, image_path1)
		endImg = warpImage(end_feat, current_feat, tri, image_path2)
		dst = cv2.addWeighted(startImg, 1-ratio, endImg, ratio, 0)
		frames.append(dst)
	return frames
		# cv2.imshow('masked image', dst)
		# cv2.waitKey()
		# cv2.destroyAllWindows()


def main(image1_path, image2_path):

	img1, gray1 = readImage(image1_path)
	img2, gray2 = readImage(image2_path)

	# load face model

	conns = facetracker.LoadCon(r'external\FaceTracker\model\face.con')
	trigs = facetracker.LoadTri(r'external\FaceTracker\model\face.tri')
	tracker1 = facetracker.FaceTracker(r'external\FaceTracker\model\face.tracker')
	tracker2 = facetracker.FaceTracker(r'external\FaceTracker\model\face.tracker')
	tracker1.setWindowSizes((11, 9, 7))
	tracker2.setWindowSizes((11, 9, 7))
	# search feature points
	
	feature_pair = getFeaturePoints(tracker1, tracker2, gray1, gray2)
	intermediate_feature = interpolatePts(feature_pair)
	frames = combineImages(intermediate_feature, TRIANGLES, image1_path, image2_path)
	frames.extend(frames[::-1])
	while(True):
		for i in range (0,len(frames)): 
			f = frames[i]
			cv2.waitKey(20) 
			cv2.imshow("Cameras",f) 
			cv2.waitKey(20)
	

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Run pyfacetrack on an image')
	parser.add_argument('image1', help='Path to image 1', default=None)
	parser.add_argument('image2', help='Path to image 2', default=None)
	args = parser.parse_args()
	
	main(args.image1, args.image2)