import argparse
import cv2
import numpy as np
import onnxruntime as ort

class PP_YOLOE_Plus():
    def __init__(self, model_path, confThreshold=0.5):
        self.classes = list(map(lambda x: x.strip(), open('coco.names', 'r').readlines()))
        self.num_class = len(self.classes)
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.session = ort.InferenceSession(model_path, so)
        model_inputs = self.session.get_inputs()
        self.input_name0 = model_inputs[0].name
        self.input_name1 = model_inputs[1].name
        self.input_shape = model_inputs[0].shape
        self.input_height = int(self.input_shape[2])
        self.input_width = int(self.input_shape[3])

        self.confThreshold = confThreshold
        self.scale_factor = np.array([[1, 1]]).astype(np.float32)

    def detect(self, srcimg):
        img = cv2.resize(cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB), (self.input_width, self.input_height))
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0

        # Inference
        results = self.session.run(None, {self.input_name0: img, self.input_name1:self.scale_factor})

        bbox, bbox_num = results
        keep_idx = (bbox[:, 1] > self.confThreshold) & (bbox[:, 0] > -1)
        bbox = bbox[keep_idx, :]
        ratioh = srcimg.shape[0] / self.input_height
        ratiow = srcimg.shape[1] / self.input_width
        for (clsid, score, xmin, ymin, xmax, ymax) in bbox:
            xmin = int(xmin * ratiow)
            ymin = int(ymin * ratioh)
            xmax = int(xmax * ratiow)
            ymax = int(ymax * ratioh)
            cv2.rectangle(srcimg, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=2)
            cv2.putText(srcimg, self.classes[int(clsid)] + ': ' + str(round(score, 2)), (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), thickness=1)
        return srcimg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelpath", type=str, default='weights/ppyoloe_plus_crn_s_80e_coco_640x640.onnx', help="model path")
    parser.add_argument("--imgpath", type=str, default='images/bus.jpg', help="image path")
    parser.add_argument("--confThreshold", default=0.5, type=float, help='class confidence')
    args = parser.parse_args()

    net = PP_YOLOE_Plus(args.modelpath, confThreshold=args.confThreshold)
    srcimg = cv2.imread(args.imgpath)
    srcimg = net.detect(srcimg)

    winName = 'Deep learning object detection in ONNXRuntime'
    cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
    cv2.imshow(winName, srcimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()