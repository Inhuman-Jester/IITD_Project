import os
import time
import cv2
import numpy as np
from PIL import Image
from FAS.PRNet_Depth_Generation.api import PRN
import FAS.PRNet_Depth_Generation.utils.depth_image as DepthImage
import tensorflow as tf
from FAS.FAS_SGTD.fas_sgtd_multi_frame.generate_network import generate_network as model_fn

class AntiSpoofing:
    def __init__(self):
        self.model = None
        self.model_checkpoint_path = "./FAS/FAS-SGTD/fas_sgtd_multi_frame/model_save/model.ckpt-19501.data-00001-of-00002" 
        self.classifier = tf.estimator.Estimator(
            model_fn=model_fn
        )    

    def predict(self, faces):
        # Placeholder implementation - replace with actual anti-spoofing logic
        image_face_cat,  vertices_map_cat, mask_cat = self.preprocess(faces)

        features = {
        "images": image_face_cat,
        "maps": vertices_map_cat,
        "masks": mask_cat,
        "labels": np.array([[0]], dtype=np.float32),   # dummy
        "names": np.array(["inference"])
        }

        def input_fn():
            dataset = tf.data.Dataset.from_tensor(features)
            return features

        features = self.classifier.predict(
            input_fn = input_fn,
            checkpoint_path = self.model_checkpoint_path
        )

        print("features: ", features)
            

    def preprocess(self, faces):
        # Placeholder implementation - replace with actual preprocessing logic
        image_face_list = []
        vertices_map_list = []
        for face in faces:
            frame = face[0]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame)
            b_box = face[1].bbox
            
            image = self.face_crop(pil_image, b_box)
            image = image.resize([256, 256])
            image = (np.array(image, np.float32) - 127.5)/255.0

            depth_map = self.generate_depth_map(pil_image, b_box)
            depth_map = depth_map.resize([32, 32])
            vertices_map = np.array(depth_map, np.float32)
            vertices_map = np.expand_dims(vertices_map, axis = -1) 

            image_face_list.append(image)
            vertices_map_list.append(vertices_map)

        image_face_cat = np.concatenate(image_face_list, axis=-1)
        vertices_map_cat = np.concatenate(vertices_map_list, axis=-1)
        mask_cat = np.array(vertices_map_cat > 0.0, np.float32) 

        image_face_cat = np.expand_dims(image_face_cat, axis=0)
        vertices_map_cat = np.expand_dims(vertices_map_cat, axis=0)
        mask_cat = np.expand_dims(mask_cat, axis=0)

        ALLDATA=[image_face_cat, vertices_map_cat, mask_cat]

        return ALLDATA 
    
    def face_crop(self, frame, b_box):
        # Placeholder implementation - replace with actual face resizing logic
        
        x1, y1, x2, y2 = [int(v) for v in b_box]
        h, w = frame.shape[:2]
        face = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

        return face
    
    def generate_depth_map(self, image, b_box):
        prn = PRN(is_dlib=False, is_opencv=False)

        pos = prn.process(image, image_info=b_box)
        kpt = prn.get_landmarks(pos)
        vertices = prn.get_vertices(pos)

        depth_map = DepthImage.generate_depth_image(vertices, kpt, image.shape, isMedFilter=True)

        return depth_map

    
    def load_model(self, model_path):
        # Placeholder implementation - replace with actual model loading logic
        self.model = None
    
    def forward_pass(self, input_data):
        # Placeholder implementation - replace with actual forward pass logic, return spoof score
        return None
    
    def decision(self, spoof_score, threshold):
        # Placeholder implementation - replace with actual decision logic
        return spoof_score > threshold
    
