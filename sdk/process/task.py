import json
import cv2
import numpy as np
from sqlalchemy.sql.functions import count
from tqdm import tqdm
from celery import group
from service.ocr import CardOcr
from commons.errors import Errors
from service.topic import Topic, Count
from service.auth import ImageAuth
from process.models import Process
from server.celery import app as c_app
from backoffice.models import Backoffice
from django.core.files.storage import default_storage
from commons.util import save_base65_file, make_directory
from commons.util import get_video_frames_v1, load_image
from base.algorithm import check_pass, check_pass_with_args
import os
import gc
import random
import torch
import copy
from time import sleep
import tensorflow as tf
from dotenv import load_dotenv
from server.settings import BASE_DIR

load_dotenv(BASE_DIR / "server/.env")


@c_app.on_after_finalize.connect
def setup_periodic_tasks(sender, **kwargs):
    sender.add_periodic_task(59.0, fetch_task.s())


topic = Topic()
counter = Count()


@c_app.task
@check_pass
def fetch_task():
    topic.fetch_data()
    processes = Process.objects.filter(process_status=Process.Status.INIT, process_type=Process.Type.AUTH)
    for p in tqdm(processes):
        topic.preprocess(p)

    try:
        if processes.count() == 0:
            torch.cuda.empty_cache()
    except:
        print("EXP")

    res = master_task.delay()


def obj(o):
    if isinstance(o, dict):
        return o
    elif isinstance(o, str):
        return json.loads(o)
    else:
        return json.loads(json.dumps(o))


def get_path_nin(process):
    return default_storage.path(process.process_id), process.process_json['nationalCode']


def get_video_frames_v2(video_path, video_len=6):
    if len(tf.config.list_physical_devices('GPU')):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(f'{len(gpus)} Physical, {len(logical_gpus)} Logical, Settings Done')

                with tf.device('/device:CPU:0'):
                    from mtcnn import MTCNN
                    detector = MTCNN()
                    capture = cv2.VideoCapture(video_path)
                    counter = 0
                    frames = []
                    boxes = []
                    faces = []
                    count = 0
                    can_rad = False
                    while capture.isOpened():
                        can_rad = True
                        ret, frame = capture.read()
                        if ret:
                            if counter > video_len:
                                count += 1

                                img_det = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                                boxes_ = detector.detect_faces(img_det)

                                if len(boxes_):
                                    for box in boxes_:

                                        x1, y1, width, height = box['box']
                                        x2, y2 = x1 + width, y1 + height
                                        face = img_det[y1:y2, x1:x2]
                                        if face.shape[0] * face.shape[1] < (120 * 120):
                                            continue

                                        boxes.append(copy.copy([x1, y1, x2, y2]))
                                        frames.append(img_det)
                                        faces.append(copy.copy(face))
                                        counter += 1


                                if counter > int(os.getenv("FUM")):
                                    break
                                if count > 35:
                                    break
                            if counter <= video_len:
                                counter += 1
                        else:
                            break

                del capture, can_rad, count, counter, detector
                gc.collect()
                return frames, faces, boxes

            except RuntimeError as e:
                print("Memory growth")
                with tf.device('/device:CPU:0'):
                    from mtcnn import MTCNN
                    detector = MTCNN()
                    capture = cv2.VideoCapture(video_path)
                    counter = 0
                    frames = []
                    boxes = []
                    faces = []
                    count = 0
                    can_rad = False
                    while capture.isOpened():
                        can_rad = True
                        ret, frame = capture.read()
                        if ret:
                            if counter > video_len:
                                count += 1

                                img_det = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                                boxes_ = detector.detect_faces(img_det)

                                if len(boxes_):
                                    for box in boxes_:

                                        x1, y1, width, height = box['box']
                                        x2, y2 = x1 + width, y1 + height
                                        face = img_det[y1:y2, x1:x2]
                                        if face.shape[0] * face.shape[1] < (120 * 120):
                                            continue

                                        boxes.append(copy.copy([x1, y1, x2, y2]))
                                        frames.append(img_det)
                                        faces.append(copy.copy(face))
                                        counter += 1

                                if counter > int(os.getenv("FUM")):
                                    break
                                if count > 35:
                                    break
                            if counter <= video_len:
                                counter += 1
                        else:
                            break

                del capture, can_rad, count, counter, detector
                gc.collect()
                return frames, faces, boxes
    else:
        with tf.device('/device:CPU:0'):
            from mtcnn import MTCNN
            detector = MTCNN()
            capture = cv2.VideoCapture(video_path)
            counter = 0
            frames = []
            boxes = []
            faces = []
            count = 0
            can_rad = False
            while capture.isOpened():
                can_rad = True
                ret, frame = capture.read()
                if ret:
                    if counter > video_len:
                        count += 1

                        img_det = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        boxes_ = detector.detect_faces(img_det)

                        if len(boxes_):
                            for box in boxes_:

                                x1, y1, width, height = box['box']
                                x2, y2 = x1 + width, y1 + height
                                face = img_det[y1:y2, x1:x2]
                                if face.shape[0] * face.shape[1] < (120 * 120):  # (100 * 200):
                                    continue
                                boxes.append(copy.copy([x1, y1, x2, y2]))
                                frames.append(img_det)
                                faces.append(copy.copy(face))
                                counter += 1

                        if counter > int(os.getenv("FUM")):  # 21 15 #41  # 60, 150, 75
                            break
                        if count > 35:  # 55
                            break
                    if counter <= video_len:
                        counter += 1
                else:
                    break

        del capture, can_rad, count, counter, detector
        gc.collect()
        return frames, faces, boxes


def get_frames(process):
    try:
        dest_path, nin = get_path_nin(process)
        frames, faces, boxes = get_video_frames_v2(f'{dest_path}/{nin}.wmv')
        return frames, faces, boxes
    except Exception as e:
        print(f"{e}")


def check_liveness(process, image_auth, frames, boxes):
    try:
        dest_path, nin = get_path_nin(process)
        res_liveness, base64_frame = image_auth.check_liveness(process, frames, boxes)
        save_base65_file(f'{dest_path}/{nin}_face.jpg', base64_frame)
        return base64_frame
    except Exception as e:
        raise e


def check_verify(process, image_auth, img, faces, is_on=False):
    try:
        res_verification = image_auth.check_verify(img, faces, process) if not is_on else image_auth.check_verify_v2(
            img,
            faces,
            process)
        return res_verification

    except Exception as e:
        raise e


@check_pass_with_args
def auth(process):
    pk = process.process_id

    if Backoffice.objects.filter(process=process, process_type=Process.Type.AUTH).count() == 0:
        backoffice, _ = Backoffice.objects.get_or_create(process=process,
                                                         process_type=Backoffice.Type.AUTH,
                                                         process_status=Backoffice.Status.INIT)
    else:
        backoffice = Backoffice.objects.get(process=process, process_type=Backoffice.Type.AUTH)
        backoffice.process_status = Backoffice.Status.INIT

    backoffice.process_status = Backoffice.Status.PENDING
    backoffice.save()

    process_err = {} if len(process.process_errors) == 0 else obj(process.process_errors)

    dest_path = default_storage.path(process.process_id)
    nin = process.process_json['nationalCode']

    img = load_image(f'{dest_path}/{nin}.jpg')

    frames, faces, boxes = get_frames(process)

    img_auth = ImageAuth()

    try:
        base64_frame = check_liveness(process, img_auth, frames, boxes)
    except Exception as e:
        process_err["live-e"] = str(e)

    try:
        res_verification = check_verify(process, img_auth, img, faces)
    except Exception as e:
        process_err["verify-e"] = str(e)

    try:
        if os.getenv("IS_ON") == '1':
            res_verification_2 = check_verify(process, img_auth, img, faces, True)
    except Exception as e:
        process_err["verify-2-e"] = str(e)

    post_urn = process.process_json['ekycDocumentPackage']['ekycVideoUrn']

    while (True):
        try:

            score = 0
            votting = 0
            desc = ''
            backoffice = img_auth.get_backoffice(process)

            # TODO ===========
            if backoffice.process_status == Process.Status.FAILED:
                print('مشکلی پیش آمده است.')

            else:

                if len(frames) == 0 or len(faces) == 0:
                    desc = Errors.FACE_NOT_FOUND.value
                    print(f'{Errors.FACE_NOT_FOUND.name}')
                else:
                    desc, votting = img_auth.is_pass_liveness(process)

                    if votting:
                        desc, votting = img_auth.is_pass_verify(process, 'verify_voting')
                        score = create_resp(res_verification, True)

                        if os.getenv("IS_ON") == '1' and votting != 1:
                            desc, votting = img_auth.is_pass_verify(process, 'verify_voting_v2')
                            score = create_resp(res_verification_2, False)

            # topic.cdn.task_complete_auth(process.process_id, post_urn, bool(votting),
            #                              base64_frame, score=score, desc=desc)

            process.process_errors = process_err
            process.save()
            break

        except Exception as e:
            process_err['task_complete'] = str(e)
            process.save()

            if '400' in str(e):
                break

            elif '401' in str(e):
                process.save()
            else:
                process.save()
                break
            sleep(0.2)

    try:
        res_verification, res_verification_2, res_liveness = img_auth.get_res(process)
    except Exception as e:
        print(e)
        res_verification = res_liveness = res_verification_2 = ''

    del frames, faces, base64_frame, res_verification, res_liveness, res_verification_2, img_auth
    gc.collect()
    return True


@check_pass_with_args
def auth_cpu(process):
    with tf.device('/device:CPU:0'):
        pk = process.process_id

        if Backoffice.objects.filter(process=process, process_type=Process.Type.AUTH).count() == 0:
            backoffice, _ = Backoffice.objects.get_or_create(process=process,
                                                             process_type=Backoffice.Type.AUTH,
                                                             process_status=Backoffice.Status.INIT)
        else:
            backoffice = Backoffice.objects.get(process=process, process_type=Backoffice.Type.AUTH)
            backoffice.process_status = Backoffice.Status.INIT

        backoffice.process_status = Backoffice.Status.PENDING
        backoffice.save()

        process_err = {} if len(process.process_errors) == 0 else obj(process.process_errors)

        dest_path = default_storage.path(process.process_id)
        nin = process.process_json['nationalCode']

        img = load_image(f'{dest_path}/{nin}.jpg')

        frames, faces, boxes = get_frames(process)

        img_auth = ImageAuth()

        try:
            base64_frame = check_liveness(process, img_auth, frames, boxes)
        except Exception as e:
            print(e)
            process_err["live-e"] = str(e)

        try:
            res_verification = check_verify(process, img_auth, img, faces)
        except Exception as e:
            print(e)
            process_err["verify-e"] = str(e)

        try:
            if os.getenv("IS_ON") == '1':
                res_verification_2 = check_verify(process, img_auth, img, faces, True)
        except Exception as e:
            print(e)
            process_err["verify-2-e"] = str(e)

        post_urn = process.process_json['ekycDocumentPackage']['ekycVideoUrn']

        while (True):
            try:

                score = 0
                votting = 0
                desc = ''
                backoffice = img_auth.get_backoffice(process)

                # TODO ===========
                if backoffice.process_status == Process.Status.FAILED:
                    print('مشکلی پیش آمده است.')

                else:

                    if len(frames) == 0 or len(faces) == 0:
                        desc = Errors.FACE_NOT_FOUND.value
                        print(f'{Errors.FACE_NOT_FOUND.name}')
                    else:
                        desc, votting = img_auth.is_pass_liveness(process)

                        if votting:
                            desc, votting = img_auth.is_pass_verify(process, 'verify_voting')
                            score = create_resp(res_verification, True)

                            if os.getenv("IS_ON") == '1' and votting != 1:
                                desc, votting = img_auth.is_pass_verify(process, 'verify_voting_v2')
                                score = create_resp(res_verification_2, False)

                # topic.cdn.task_complete_auth(process.process_id, post_urn, bool(votting),
                #                              base64_frame, score=score, desc=desc)


                process.process_errors = process_err
                process.save()
                break

            except Exception as e:
                print(e)
                process_err['task_complete'] = str(e)
                process.save()

                if '400' in str(e):
                    break

                elif '401' in str(e):
                    process.save()
                else:
                    process.save()
                    break
                sleep(0.2)

        try:
            res_verification, res_verification_2, res_liveness = img_auth.get_res(process)
        except Exception as e:
            res_verification = res_liveness = res_verification_2 = ''

        del frames, faces, base64_frame, res_verification, res_liveness, res_verification_2, img_auth
        gc.collect()
        return True


@check_pass_with_args
def ocr(process):
    ocr_card = CardOcr(topic.cdn)
    dest_path = default_storage.path(process.process_id)
    make_directory(dest_path)
    nin = process.process_json['nationalCode']
    national_card_image_front = process.process_json['ekycDocumentPackage']['nationalCardImageFrontUrn']
    national_card_image_back = process.process_json['ekycDocumentPackage']['nationalCardImageBackUrn']
    is_pass = True
    res = ''
    process_url = {}
    process_err = {}

    try:
        process_url = {} if len(process.process_url) == 0 else obj(process.process_url)
        process_err = {} if len(process.process_errors) == 0 else obj(process.process_errors)
    except Exception as e:
        process_err['ocr-init'] = str(e)
        print(e)

    try:
        ocr_card.save_file(f"{dest_path}/{nin}_nationalCardImageFront.jpg", national_card_image_front)
        process_url[
            'nationalCardImageFront'] = f'{default_storage.url(process.process_id)}/{nin}_nationalCardImageFront.jpg'

    except Exception as e:
        print(e)
        process_err[national_card_image_front] = str(e)
        # topic.cdn.task_complete_ocr(process.process_id, national_card_image_front, False)
        is_pass = False

    try:
        ocr_card.save_file(f"{dest_path}/{nin}_nationalCardImageBack.jpg", national_card_image_back)
        process_url[
            'nationalCardImageBack'] = f'{default_storage.url(process.process_id)}/{nin}_nationalCardImageBack.jpg'

    except Exception as e:
        print(e)
        process_err[national_card_image_back] = str(e)
        is_pass = False

    if os.getenv("OCR") == '1':
        try:
            res = ocr_card.back_card(process, f"{dest_path}/{nin}_nationalCardImageBack.jpg")
            # topic.cdn.task_complete_ocr(process.process_id, national_card_image_back, True, res)
        except Exception as e:
            print(e)
            process_err['ocr-back-image-error'] = str(e)
            try:
                print(f'{process.process_id}')
            except Exception as e:
                print(e)

        try:
            res = ocr_card.front_card(process, f"{dest_path}/{nin}_nationalCardImageFront.jpg")

        except Exception as e:
            print(e)
            process_err['ocr-front-image-error'] = str(e)
            try:
                print(f'{process.process_id}')
            except Exception as e:
                print(e)

    process.process_url = json.dumps(process_url)
    process.process_errors = process_err
    process.save()
    del ocr_card
    gc.collect()


@check_pass_with_args
def ocr_cpu(process):
    with tf.device('/device:CPU:0'):

        ocr_card = CardOcr(topic.cdn)
        dest_path = default_storage.path(process.process_id)
        make_directory(dest_path)
        nin = process.process_json['nationalCode']
        national_card_image_front = process.process_json['ekycDocumentPackage']['nationalCardImageFrontUrn']
        national_card_image_back = process.process_json['ekycDocumentPackage']['nationalCardImageBackUrn']
        is_pass = True
        res = ''
        process_url = {}
        process_err = {}

        try:
            process_url = {} if len(process.process_url) == 0 else obj(process.process_url)
            process_err = {} if len(process.process_errors) == 0 else obj(process.process_errors)
        except Exception as e:
            process_err['ocr-init'] = str(e)
            print(e)
        try:
            ocr_card.save_file(f"{dest_path}/{nin}_nationalCardImageFront.jpg", national_card_image_front)
            process_url[
                'nationalCardImageFront'] = f'{default_storage.url(process.process_id)}/{nin}_nationalCardImageFront.jpg'
        except Exception as e:
            print(e)
            process_err[national_card_image_front] = str(e)
            is_pass = False

        try:
            ocr_card.save_file(f"{dest_path}/{nin}_nationalCardImageBack.jpg", national_card_image_back)
            process_url[
                'nationalCardImageBack'] = f'{default_storage.url(process.process_id)}/{nin}_nationalCardImageBack.jpg'
        except Exception as e:
            print(e)
            process_err[national_card_image_back] = str(e)
            is_pass = False

        if os.getenv("OCR") == '1':
            try:
                res = ocr_card.back_card(process, f"{dest_path}/{nin}_nationalCardImageBack.jpg")
            except Exception as e:
                process_err['ocr-back-image-error'] = str(e)
                try:
                    print(f'{process.process_id}')
                except Exception as e:
                    print(e)

            try:
                res = ocr_card.front_card(process, f"{dest_path}/{nin}_nationalCardImageFront.jpg")
            except Exception as e:
                print(e)
                process_err['ocr-front-image-error'] = str(e)
                try:
                    print(f'{process.process_id}')
                except Exception as e:
                    print(e)

        process.process_url = json.dumps(process_url)
        process.process_errors = process_err
        process.save()
        del ocr_card
        gc.collect()


@c_app.task()
def slave_task(pk, use_gpu):
    process = Process.objects.filter(process_pk=pk).first()
    process.process_status = Process.Status.PENDING
    process.process_try += 1
    process.save()
    rnd = random.random()
    try:
        if use_gpu:
            auth(process)
            ocr_cpu(process)
        else:
            auth_cpu(process)
            ocr_cpu(process)

        process.process_status = Process.Status.COMPLETED
        process.save()

    except Exception as e:
        process.process_status = Process.Status.ERROR
        process.process_try += 1
        process.save()

    if use_gpu:
        counter.update_m(0)
    else:
        counter.update_m(1)

    return True


@check_pass_with_args
def change_task_status(pr):
    for p in pr:
        p.process_status = Process.Status.INIT
        p.save()


@c_app.task(bind=True)
def master_task(self):
    if len(tf.config.list_physical_devices('GPU')):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)
                logical_gpus = tf.config.list_logical_devices('GPU')
            except RuntimeError as e:
                print("Memory growth")
    else:
        processes = Process.objects.exclude(process_type=Process.Type.OCR) \
        .exclude(process_status=Process.Status.COMPLETED) \
        .exclude(process_status=Process.Status.FAILED) \
        .exclude(process_status=Process.Status.PENDING) \
        .exclude(process_status=Process.Status.ERROR) \
        # .exclude(process_try__gte=3)

    list_jobs = []
    for process in processes:
        if len(tf.config.list_physical_devices('GPU')) and os.getenv("ENABLED_G") == '1':
            if counter.get_val(0) < int(os.getenv("USED")):
                list_jobs.append(slave_task.s(process.pk, True))
                counter.update_a(0)
            else:
                list_jobs.append(slave_task.s(process.pk, False))
                counter.update_a(1)
        else:
            list_jobs.append(slave_task.s(process.pk, False))
            counter.update_a(1)

    jobs = group(list_jobs)
    jobs.delay()
    del list_jobs
    gc.collect()
    return True


def create_resp(res, step=True):
    if step == True:
        if np.average(res['distance']) <= 0.40:
            res = 100 - 10 * np.average(res['distance'])
            return float(f"{res:.2f}")
        res = (1 - np.average(res['distance'])) * 100
        return float(f"{res:.2f}")
    else:
        res = np.average(res['distance']) * 100
        return float(f"{res:.2f}")
