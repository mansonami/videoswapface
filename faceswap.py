# -*- coding: utf-8 -*-  
  
import cv2  
from imutils import video
import dlib
import face_recognition
import argparse
import glob
import os
import numpy
import math

    
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

SCALE_FACTOR = 1
FEATHER_AMOUNT = 11

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                               RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS + FACE_POINTS)
# Points from the second image to overlay on the first. The convex hull of each
# element will be overlaid.
OVERLAY_POINTS = [LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                               RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS + FACE_POINTS
]


# Amount of blur to use during colour correction, as a fraction of the
# pupillary distance.
COLOUR_CORRECT_BLUR_FRAC = 0.6

facial_features = [
    'chin',
    'right_eyebrow',
    'left_eyebrow',
    'nose_bridge',
    'nose_tip',
    'right_eye',
    'left_eye',
    'top_lip',
    'bottom_lip'
]

def correct_colours(im1, im2, landmarks1):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * numpy.linalg.norm(
                              numpy.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                              numpy.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(numpy.float64) * im1_blur.astype(numpy.float64) /
                                                im2_blur.astype(numpy.float64))

def warp_im(im, M, dshape):
    output_im = numpy.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)

    return output_im

def transformation_from_points(points1, points2):
    """
    Return an affine transformation [s * R | T] such that:

        sum ||s*R*p1,i + T - p2,i||^2

    is minimized.

    """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)
    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)

    points1 -= c1
    points2 -= c2

    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = numpy.linalg.svd(points1.T * points2)


    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T

    return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         numpy.matrix([0., 0., 1.])])

def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    #print(im.shape, points.shape)
    cv2.fillConvexPoly(im, points, color=color)

def get_face_mask(im, landmarks):
    im = numpy.zeros(im.shape[:2], dtype=numpy.float64)

    for group in OVERLAY_POINTS:
        #print("group :", group, len(group))
        draw_convex_hull(im,
                         landmarks[group],
                         color=1)

    #im = numpy.array([im, im, im]).transpose((1, 2, 0))
    im = numpy.array([im, im, im]).transpose((1, 2, 0))

    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im

def face_orientation(frame, landmarks):
    size = frame.shape #(height, width, color_channel)
                        
    image_points = numpy.array([
                            landmarks[31],     # Nose tip
                            landmarks[8],   # Chin
                            landmarks[45],     # Left eye left corner
                            landmarks[36],     # Right eye right corne
                            landmarks[54],     # Left Mouth corner
                            landmarks[48]      # Right mouth corner
                        ], dtype="double")
                        
    # image_points = numpy.array([
        # [landmarks[33, 0], landmarks[33, 1]],     # Nose tip
        # (landmarks[8, 0], landmarks[8, 1]),       # Chin
        # (landmarks[36, 0], landmarks[36, 1]),     # Left eye left corner
        # (landmarks[45, 0], landmarks[45, 1]),     # Right eye right corner
        # (landmarks[48, 0], landmarks[48, 1]),     # Left Mouth corner
        # (landmarks[54, 0], landmarks[54, 1])      # Right mouth corner
        # ], dtype="double")

                        
    model_points = numpy.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-165.0, 170.0, -135.0),     # Left eye left corner
                            (165.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner                         
                        ])

    # Camera internals
 
    center = (size[1]/2, size[0]/2)
    focal_length = center[0] / numpy.tan(60/2 * numpy.pi / 180)
    camera_matrix = numpy.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )

    dist_coeffs = numpy.zeros((4,1)) # Assuming no lens distortion
    #(success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.CV_ITERATIVE)
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    
    axis = numpy.float32([[500,0,0], 
                          [0,500,0], 
                          [0,0,500]])
                          
    imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    modelpts, jac2 = cv2.projectPoints(model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]

    proj_matrix = numpy.hstack((rvec_matrix, translation_vector))
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6] 

    
    pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]


    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))

    #return imgpts, modelpts, (str(int(roll)), str(int(pitch)), str(int(yaw))), (landmarks[4], landmarks[5])
    return numpy.array([float(roll), float(pitch), float(yaw)], dtype = "double")

def get_key(dict, value):
    for key in dict:
        if((dict[key] == value).all()):
            return key

def getclose(A, B):
    A = numpy.array(list(A))
    B = numpy.array(list(B.values()))
    # distances array-wise
    numpy.abs(B - A)

    # sum of absolute values of distances (smallest is closest)
    numpy.sum(numpy.abs(B - A), axis=1)

    # index of smallest (in this case index 1)
    numpy.argmin(numpy.sum(numpy.abs(B - A), axis=1))

    # all in one line (take array 1 from B)
    result = B[numpy.argmin(numpy.sum(numpy.abs(B - A), axis=1))]
    
    pic = get_key(posedict, result)
    
    return pic

def get_landmarks(im):
    rects = detector(im, 1)
    
    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces

    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])
    
def read_im_and_landmarks(fname):
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                         im.shape[0] * SCALE_FACTOR))
    s = get_landmarks(im)

    return im, s
    
def loadpose(path):
    print("Load the face pose")
    input_paths = glob.glob(os.path.join(path, "*.jpg"))
    if len(input_paths) == 0:
        input_paths = glob.glob(os.path.join(path, "*.png"))
    for pic in input_paths:
        try:
            im, landmarks = read_im_and_landmarks(pic)
            #print(pic)
            posedict[pic]  = face_orientation(im, landmarks)
        except:
            continue
            
    return posedict

def getlmface(frame, star):
    # Load a sample picture and learn how to recognize it.
    star_image = face_recognition.load_image_file(star)
    star_face_encoding = face_recognition.face_encodings(star_image)[0]

    # Create arrays of known face encodings and their names
    known_face_encodings = [
        star_face_encoding
    ]

    # Find all the faces and face encodings in the unknown image
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    i = 0
    landmarks_org = []
    # Loop through each face found in the unknown image
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = []
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)

        # If a match was found in known_face_encodings, just use the first one.
        # return the same pose face pic
        if True in matches:

            #first_match_index = matches.index(True)
            #name = known_face_names[first_match_index]
            face_landmarks_list = face_recognition.face_landmarks(frame)  # don't know to get the match one with parament?            
            face_landmarks = face_landmarks_list[i]
            for tmp in facial_features:
                if(tmp == "top_lip"):
                    landmarks_org.extend(face_landmarks[tmp][:8])
                else:
                    landmarks_org.extend(face_landmarks[tmp])
                #print(landmarks_org, len(landmarks_org))
            #rep_pic = getclose(face_orientation(frame, landmarks_org), posedict)
            #print(rep_pic, face_orientation(frame, numpy.array(landmarks_org)), posedict[rep_pic])
        break
        
    return frame, landmarks_org
    

def main(): 
    # video format
    videoCapture = cv2.VideoCapture(args.video)
    fps = int(videoCapture.get(5))

      
    # get size
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
      
    #special format, I420-avi, MJPG-mp4  
    videoWriter = cv2.VideoWriter('oto_other.mp4', cv2.VideoWriter_fourcc(*'XVID'), fps, size)  
      
    #read
    success, frame = videoCapture.read()  
      
    while success :  
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]
        
        # The frame has the face in it
        rects = detector(rgb_frame, 1)
        if(len(rects) > 0):
            print("The frame has face...")
            # return the star face landmarks
            rgb_frame, rgb_landmark = getlmface(rgb_frame, args.pic)
            if(len(rgb_landmark) > 0):
                print("The frame has star face...")
                rep_pic = getclose(face_orientation(rgb_frame, rgb_landmark), posedict)
                
                im2, landmarks2 = read_im_and_landmarks(rep_pic)

                
                # get the affix matrix
                rgb_landmark = numpy.matrix(rgb_landmark)
                #im2 = im2[:, :, ::-1]
                #print(landmarks2[ALIGN_POINTS], rgb_landmark[ALIGN_POINTS])
                M = transformation_from_points(rgb_landmark[ALIGN_POINTS], landmarks2[ALIGN_POINTS])
                
                mask = get_face_mask(im2, landmarks2)            
                warped_mask = warp_im(mask, M, rgb_frame.shape)
                
                combined_mask = numpy.max([get_face_mask(rgb_frame, rgb_landmark), warped_mask], axis=0)            
                warped_im2 = warp_im(im2, M, rgb_frame.shape)            
                warped_corrected_im2 = correct_colours(rgb_frame, warped_im2, rgb_landmark)
                output_im = rgb_frame * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
                #cv2.imwrite("outputbef.png", output_im)
                frame = output_im[:, :, ::-1]
                #cv2.imwrite("outputaf.png", frame)
        else:
            frame = rgb_frame[:, :, ::-1]
        
        # show video
        #cv2.imshow("Oto Video", frame)
        # delay
        cv2.waitKey(int(1000/int(fps)))
        # generate video
        videoWriter.write(numpy.uint8(frame))
        # next frame
        success, frame = videoCapture.read()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', dest='video', type=str, help='Org video need to be edited')
    parser.add_argument('--replace', dest='replace', type=str, help='The wanted star path of *.png or *.jpg')
    parser.add_argument('--pic', dest='pic', type=str, help='The guy who should be replaced in the video')
    args = parser.parse_args()

    
    # load the replace-face pose
    posedict = {}
    loadpose(args.replace)
    print(posedict)

    main()