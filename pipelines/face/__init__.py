


from pipelines.face.detection import FaceDetector


class FacePipeline:
    def __init__(self, aligment = False):
        self.face_detector = FaceDetector()