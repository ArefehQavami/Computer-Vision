from commons_service import VerificationModel


class FaceVerification:
    def __init__(self):
        self.model = VerificationModel()

    def predict(self, image_A, image_B):
        res = {'ans': self.model.verify_face(image_A, image_B, is_face=True)}
        return res

    def pridect_with_frames(self, image_source, frames):
        return self.model.verify_face_frames(image_source, frames, is_face=True)
