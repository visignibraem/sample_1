import logging
from research.model_compare import *
from tools.inference import get_detector_parameters

"""
   Загрузка моделей:
       Detector: - для детекторов tflite (не проверяется точность, здесь он только для получения кропов)
       DefaultModel: - для классификаторов tflite (тестировалось только с SKU, не с пустотами или posm)
       EfficientNetModel: - для классификаторов .h5 Efficientnet. С метрикой. (для работы нужен env с установленным \
       efficientnet. см. /home/visignibraem/efficientnetenv/

       clarify_skuids_by_size TODO: сейчас не работает!

   ModelCompare:
       Класс имеет несколько вариантов работы.
       1) Сравнение двух моделей
       2) Сравнение модели и gt
       Параметр model_test обязателен - объект тестируемой модели.
       Параметр model_true опционален - модель выступающая, как GT
       Параметр img_dir опционален, используется при получении кропов детектором.  

       Загрузка данных происходит через параметр ModelCompare.data , либо используя один из методов:
       - ModelCompare.load_data(x) - где Х путь к npy массиву прочитанных изображений 
       - ModelCompare.load_data_from_detector(self, detector) - здесь податется объект детектор. \
        Указав в init ModelCompare img_dir="/path" детектор прочитает все изображения \
        из директории, создаст кропы и отдаст их классификатору.
       - ModelCompare.load_data_from_img_folder - для чтения папки с кропами.

       ModelCompare.run_on_images - Основной метод, делает отчёт по точности для каждой \
       отдельной фотографии, а затем выводит средний результат, который логгируется в log_path.
       - имеет опциональный параметр val_y, если он не None, то сравнение будет проходить не между моделями, а 
       между model_test и GT, иначе, между model_test и model_true.

       ModelCompare.evaluate - проводит обычный замер точности по всему датасету, в точности так, как это происходит \
       в calc_acc. (Не по изображениям раздельно, а по классам из полного набора данных) а вход принимает GT. 

       ModelCompare.run_on_data - проводит обычный замер точности по всему датасету, в точности так, как это происходит \
       в calc_acc. (Не по изображениям раздельно, а по классам из полного набора данных) \
       Использует в качестве GT - true_model 

   Вспомогательные методы:
       - LoadReport: парсинг выгруженного отчёта по точности, получение GT и PRED продовой модели.
       - crop_gt_from_imgs: получение кропов из разметки
       - revert_clarify: исправляет gt так, чтобы в нём были только родительские классы. \
         Для тестирования классификации без уточнения.


   """

"""
Сценарий описанный кодом ниже:
- Загружаем отчет по точности
- две модели классификатора, большую и маленькую.

1) Замер точности маленькая модель без уточнения vs GT без уточнения (чистая точность классификатора)
2) Замер точности маленькая модель без уточнения vs GT с товарами, требующими уточнения. 
3) Замер точности маленькая модель без уточнения vs Большая модель без уточнения.
4) Замер точности маленькая модель с уточнением vs GT с товарами, требующими уточнения. 
5) Замер точности маленькая модель с уточнением vs Большая модель с уточнением. 

"""


log_path = "/home/visignibraem/model_compare_prod_2.log" # TODO: create logger setup
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.FileHandler(log_path, 'w', 'utf-8')  # TODO: make csv logging for stats, etc. Export to mlflow.
formatter = logging.Formatter('%(message)s')  #
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(logging.StreamHandler())

from tools.utils import enableGPU, allow_growth

enableGPU(0)
allow_growth()

# Load GT
xml_path = "/home/ml/share/XML_F52E2B61-18A1-11d1-B105-00805F49916B1.xml"
img_dir = "/home/ml/share/de/"

report = LoadReport(xml_path, img_dir)
bboxes, skuids, paths = report.gt()
# bboxes, skuids, paths = bboxes[0:5], skuids[0:5], paths[0:5]  # crop data for fast debugging

crops, img_shapes = crop_gt_from_imgs(paths, bboxes)
skuids_no_clarify = revert_clarify(company_name='XXX', skuids=skuids)

# Загрузка моделей
small_classifier_model = "/home/ml/models/mlflow-artifact/4/215f65b93b4e45358114c3e833887804/artifacts/model_epoch-00494_loss-0.0096_val_loss-0.0525.tflite"
classifier_model = '/home/ml/models/mlflow-artifact/4/a3c4de55e21242f090c3b6950fe7ebbe/artifacts/model_epoch-00266_loss-0.0019_val_loss-0.0282.h5'
product_pricetag_model_path = '/home/ml/models/mlflow-artifact/36/1744f96740594f86916305a72443b512/artifacts/model_epoch-00146_inference.tflite'
product_pricetag_confidence_thresh, product_pricetag_iou_thresh = get_detector_parameters(product_pricetag_model_path)

# сейчас и здесь детектор не нужен, загрузка для примера
detector = Detector(product_pricetag_model_path, product_pricetag_confidence_thresh, product_pricetag_iou_thresh)

tested_model = DefaultModel(small_classifier_model,
                            small_classifier_model + '_triplet.npy', clarify_skuid=False)  # Без уточнения

big_model = EfficientNetModel(classifier_model,
                              classifier_model + '_triplet.npy', clarify_skuid=False)  # Без уточнения

mc = ModelCompare(model_true=big_model, model_test=tested_model)
mc.data = crops
# files передается опционально, когда нет imdir (работаем без детектора), для отображения имен файлов.
mc.files = paths

# Сравнение маленькой модели без уточнение по GT без уточнения
logger.info("\nSmall no clarify vs GT no clarify only parents (clean classification quality)")
mc.run_on_images(val_y=skuids_no_clarify)

# Сравнение маленькой модели без уточнение по GT с уточнением
logger.info("\nSmall no clarify vs GT with childrens.")
mc.run_on_images(val_y=skuids)

# Сравнение маленькой модели без уточнения с pred большой модели без уточнения
logger.info("\nSmall no clarify vs Big no clarify.")
mc.run_on_images()

# Загрузка моделей с уточнением
tested_model = DefaultModel(small_classifier_model,
                            small_classifier_model + '_triplet.npy',
                            clarify_skuid=True,
                            img_shapes=img_shapes,
                            bboxes=bboxes,
                            skuids_with_same_designs=skuids_with_same_designs)

big_model = EfficientNetModel(classifier_model,
                              classifier_model + '_triplet.npy',
                              clarify_skuid=True,
                              img_shapes=img_shapes,
                              bboxes=bboxes,
                              skuids_with_same_designs=skuids_with_same_designs)
# подмена моделей
mc.model_true = big_model
mc.model_test = tested_model

# Сравнение маленькой модели c уточнением с GT
logger.info("\nSmall with clarify vs GT with childrens.")
mc.run_on_images(skuids)

# Сравнение маленькой модели c уточнением с большой моделью с уточнением
logger.info("\nSmall with clarify vs Big with clarify.")
mc.run_on_images()



# Доп. Сценарий генерация кропов
x = ModelCompare(model_true=None, model_test=None, img_dir='/home/visignibraem/realogram')
x.load_data_from_detector(detector)
np.save('test_data.npy', x.data)

