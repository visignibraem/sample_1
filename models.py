from SkynetCV.SkynetCV import AlignMode
from tensorflow.keras.models import load_model

from keras_loss_function.keras_ssd_loss import *
from tools.inference import infer_detection, infer_triplet_classification, crop_images_from_pred


class MainModel:
    def __init__(self, model_path, metric_path=None, clarify_skuid=False,
                 skuids_with_same_designs=None, bboxes=None, img_shapes=None):
        self.model_path = model_path
        self.metric_path = metric_path
        self.model = None

        self.clarify_skuid = clarify_skuid
        if clarify_skuid:
            assert skuids_with_same_designs
            assert bboxes
            assert img_shapes

        self.img_shapes = img_shapes
        self.skuids_with_same_designs = skuids_with_same_designs
        self.bboxes = bboxes

    def infer_method(self, x):
        raise NotImplementedError

    def predict(self, i, x):
        y = self.infer_method(x)
        if self.clarify_skuid:
            y = self.clarify(i, y)
        return y

    def _load(self):
        if self.model_path.endswith('tflite'):
            self.model = self.model_path
        else:
            K.clear_session()
            self.model = load_model(self.model_path, custom_objects=self._custom_objects())

    def _custom_objects(self):
        return {'focal_triplet_loss': focal_triplet_loss}

    def clarify(self, i, class_id):
        """
            pred_bboxes: - list like [[skuid, conf, xmin, ymin, xmax, ymax], [skuid, conf, xmin...
            pred_sku_ids: - list like [1122, 1141, 1131]
            sku_dict: - dict like {1131: [[1131, width, height], [554, width, height]]}

            Returns:- info,  - debugging and stats info
                      pred_sku_ids - clarified sku ids
        """
        raise DeprecationWarning

        info, class_id = clarify_skus_by_size(class_id,
                                              self.bboxes[i],
                                              self.skuids_with_same_designs,
                                              self.img_shapes[i][:2])
        return class_id


class Detector(MainModel):
    def __init__(self, model_path, product_pricetag_confidence_thresh, product_pricetag_iou_thresh):
        super().__init__(model_path)
        self.product_pricetag_iou_thresh = product_pricetag_iou_thresh
        self.product_pricetag_confidence_thresh = product_pricetag_confidence_thresh

    def _load(self):
        raise NotImplementedError

    def infer_method(self, x):
        pred_y = infer_detection(x, self.model_path,
                                 confidence_thresh=self.product_pricetag_confidence_thresh,
                                 iou_thresh=self.product_pricetag_iou_thresh,
                                 keep_original_aspect_ratio=True,
                                 align_mode=AlignMode.CENTER,
                                 decode_fast=False)
        return pred_y

    def products(self, imgs):
        product_pred = self.predict(imgs)
        product_pred = [bboxs[bboxs[:, 0] == 1] if len(bboxs) else [] for bboxs in product_pred]
        product_images = crop_images_from_pred(imgs, product_pred)
        return product_images


class DefaultModel(MainModel):
    def infer_method(self, x):
        y_pred = infer_triplet_classification(x, self.model,
                                              triplet_metric_filename=self.metric_path,
                                              custom_objects=self._custom_objects(),
                                              keep_original_aspect_ratio=False,
                                              align_mode=AlignMode.LEFTTOP,
                                              efficientnet=False)
        return y_pred


class EfficientNetModel(MainModel):
    def infer_method(self, x):
        y_pred = infer_triplet_classification(x, self.model,
                                              triplet_metric_filename=self.metric_path,
                                              custom_objects=self._custom_objects(),
                                              keep_original_aspect_ratio=False,
                                              align_mode=AlignMode.LEFTTOP,
                                              efficientnet=True)
        return y_pred
