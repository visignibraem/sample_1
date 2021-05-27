import logging
import time
from multiprocessing.dummy import Pool

import numpy as np
import os
from SkynetCV import SkynetCV
from joblib import Parallel, delayed

from research.model_compare.models import DefaultModel, EfficientNetModel
from research.model_compare.webreport_parser import LoadReport
from tools.image_utils import is_image
from tools.inference import calc_metrics

logger = logging.getLogger(__name__)


class ModelCompare:
    def __init__(self, model_test, model_true=None, imdir=None):
        """
        Parameters
        ----------
        model_true MainModel. Pred is y_true
        model_test MainModel. Pred is y_pred
        """
        self.model_test = model_test
        self.model_true = model_true
        self.imdir = imdir
        self.data = None
        self.report_data = None
        self.files = None
        self.acc_df = None
        if imdir:
            self.files = sorted([os.path.join(r, f) for r, ds, fs in os.walk(imdir) for f in fs if is_image(f)])

    def run_on_images(self, val_y=None, test_true_model=False, experiment_name=None):
        self.model_test._load()
        try:
            y_pred = Parallel(n_jobs=15)(delayed(self.model_test.predict)(i, img) for i, img in enumerate(self.data))
        except TypeError as e:
            logger.fatal("Load data first! (From detector, manually et.c.) look in example.py")
            raise e

        if val_y:
            logger.debug("Val_y data exists, big model (gt model) will not be used.")
            y_true = val_y
            if test_true_model:  # test true (big) model vs gt from val_y
                self.model_true._load()
                y_pred = Parallel(n_jobs=15)(
                    delayed(self.model_true.predict)(i, img) for i, img in enumerate(self.data))
        else:
            self.model_true._load()
            y_true = Parallel(n_jobs=15)(delayed(self.model_true.predict)(i, img) for i, img in enumerate(self.data))
        self.calc_acc(y_true=y_true, y_pred=y_pred, experiment_name=experiment_name)

    def run_on_data(self):
        y_pred = self.model_test.predict(self.data)
        y_true = self.model_true.predict(self.data)
        metrics_sum, metrics_per_classes = calc_metrics(y_true, y_pred)
        return metrics_sum, metrics_per_classes

    def evaluate(self, val_y):
        assert len(val_y) == len(self.data)
        y_pred = self.model_test.predict(self.data)
        metrics_sum, metrics_per_classes = calc_metrics(val_y, y_pred)
        return metrics_sum, metrics_per_classes

    def load_data(self, path):
        self.data = np.load(path)

    def load_report_data(self, company_name, xml_path, img_dir, guids=None, only_clf_errors=False, count=-1):
        report = LoadReport(company_name=company_name,
                            xml_path=xml_path,
                            img_dir=img_dir,
                            guids=guids,
                            only_clf_errors=only_clf_errors)

        self.report_data = report.data()
        self.acc_df = report.acc_df().iloc[:count]

    def acc_df_to_excel(self, output_path):
        self.acc_df.to_excel(output_path)

    def load_data_from_img_folder(self, path):
        self.data = self.read_img_folder(path)

    def load_data_from_detector(self, detector):
        if not self.imdir:
            print("imdir path not in ModelCopmare init. Check it & run again")
        else:
            self.data = detector.products(self.read_img_folder(self.imdir))

    @staticmethod
    def read_img_folder(path):
        files = sorted([os.path.join(r, f) for r, ds, fs in os.walk(path) for f in fs if is_image(f)])
        t1 = time.time()
        with Pool(os.cpu_count()) as p:
            images = p.map(SkynetCV.load, files)
        t2 = time.time()
        logger.debug(f'Time: {t2 - t1}')
        logger.debug(f'Images count: {len(images)}')
        return images
