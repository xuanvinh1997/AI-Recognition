import os
from pipelines.face.detection.scrfd.torch2onnx import pytorch2onnx

from tools.download import download_large_file
from huggingface_hub import hf_hub_download

REPO_ID = "vinhpx/insight_face"
FILENAME = "scrfd/models/scrfd_10g.pth"


class FaceDetector:
    def __init__(self, model='scrfd_10g') -> None:
        if self.__check_model_exists(model):
            self.model_name = model
            # self.model =
            self.__convert_to_onnx()
        else:
            raise Exception(f"Model {model} does not exist")
        pass

    def __check_model_exists(self, model):
        if model in ['scrfd_10g', 'scrfd_500m']:
            # check model in local path
            if model == 'scrfd_10g':
                self.model_path = os.path.join(os.path.dirname(
                    __file__), 'scrfd', 'models', model+'.onnx')

                # check exists
                if not os.path.exists(self.model_path):
                    # download  and copy to local path
                    self.model_path = hf_hub_download(
                        repo_id=REPO_ID, filename=FILENAME)
                    # os.makedirs(os.path.dirname(model), exist_ok=True)

            return True
        elif not os.path.exists(model):
            raise Exception(f"Model {model} does not exist")

    def __convert_to_onnx(self):
        # Export the PyTorch model to ONNX format
        onnx_filename = os.path.join(os.path.dirname(
            __file__), 'scrfd', 'models', self.model_name+'.onnx')
        config_filename = os.path.join(os.path.dirname(
            __file__), 'scrfd', 'configs', self.model_name+'.py')
        # print(45, config_filename)
        pytorch2onnx(
            config_path=config_filename,
            checkpoint_path=self.model_path,
            input_img="data/test/face_detection.jpg",
            input_shape=(1, 3, 640, 640),
            opset_version=11,
            show=False,
            output_file=onnx_filename,
            normalize_cfg={'mean': [127.5, 127.5, 127.5],
                           'std': [128.0, 128.0, 128.0]},
            verify=False,
            simplify=True,
        )

        pass
