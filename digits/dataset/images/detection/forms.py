# Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os.path
import requests

import wtforms
from wtforms import validators

from ..forms import ImageDatasetForm
from digits import utils
from digits.utils.forms import validate_required_iff, validate_greater_than, validate_required_if_set


class ObjectDetectionDatasetForm(ImageDatasetForm):
    """
    Defines the form used to create a new ObjectDetectionDatasetJob
    """

    backend = wtforms.SelectField('DB backend',
                                  choices=[
                                      ('lmdb', 'LMDB'),
                                      ('hdf5', 'HDF5')
                                  ],
                                  default='lmdb',
                                  )

    def validate_backend(form, field):
        if field.data == 'lmdb':
            form.compression.data = 'none'
        elif field.data == 'tfrecords':
            form.compression.data = 'none'
        elif field.data == 'hdf5':
            form.encoding.data = 'none'

    compression = utils.forms.SelectField(
        'DB compression',
        choices=[
            ('none', 'None'),
            ('gzip', 'GZIP'),
        ],
        default='none',
        tooltip=('Compressing the dataset may significantly decrease the size '
                 'of your database files, but it may increase read and write times.'),
    )

    # Use a SelectField instead of a HiddenField so that the default value
    # is used when nothing is provided (through the REST API)
    method = wtforms.SelectField(u'Dataset type',
                                 choices=[
                                     ('folder', 'Folder'),
                                     ('textfile', 'Textfiles'),
                                     ('s3', 'S3'),
                                 ],
                                 default='folder',
                                 )

    def validate_folder_path(form, field):
        if not field.data:
            pass
        elif utils.is_url(field.data):
            # make sure the URL exists
            try:
                r = requests.get(field.data,
                                 allow_redirects=False,
                                 timeout=utils.HTTP_TIMEOUT)
                if r.status_code not in [requests.codes.ok, requests.codes.moved, requests.codes.found]:
                    raise validators.ValidationError('URL not found')
            except Exception as e:
                raise validators.ValidationError('Caught %s while checking URL: %s' % (type(e).__name__, e))
            else:
                return True
        else:
            # make sure the filesystem path exists
            # and make sure the filesystem path is absolute
            if not os.path.exists(field.data) or not os.path.isdir(field.data):
                raise validators.ValidationError('Folder does not exist')
            elif not os.path.isabs(field.data):
                raise validators.ValidationError('Filesystem path is not absolute')
            else:
                return True

    #
    # Method - folder
    #

    train_image_folder = utils.forms.StringField(
        u'Training image folder',
        validators=[
            validators.DataRequired(),
            validate_folder_path,
        ],
        tooltip="Indicate a folder of images to use for training"
    )

    train_label_folder = utils.forms.StringField(
        u'Training label folder',
        validators=[
            validators.DataRequired(),
            validate_folder_path,
        ],
        tooltip="Indicate a folder of training labels"
    )

    val_image_folder = utils.forms.StringField(
        u'Validation image folder',
        validators=[
            validate_required_if_set('val_label_folder'),
            validate_folder_path,
        ],
        tooltip="Indicate a folder of images to use for training"
    )

    val_label_folder = utils.forms.StringField(
        u'Validation label folder',
        validators=[
            validate_required_if_set('val_image_folder'),
            validate_folder_path,
        ],
        tooltip="Indicate a folder of validation labels"
    )

    resize_image_width = utils.forms.IntegerField(
        u'Resize Image Width',
        validators=[
            validate_required_if_set('resize_image_height'),
            validators.NumberRange(min=1),
        ],
        tooltip="If specified, images will be resized to that dimension after padding"
    )

    resize_image_height = utils.forms.IntegerField(
        u'Resize Image Height',
        validators=[
            validate_required_if_set('resize_image_width'),
            validators.NumberRange(min=1),
        ],
        tooltip="If specified, images will be resized to that dimension after padding"
    )

    padding_image_width = utils.forms.IntegerField(
        u'Padding Image Width',
        default=1248,
        validators=[
            validate_required_if_set('padding_image_height'),
            validators.NumberRange(min=1),
        ],
        tooltip="If specified, images will be padded to that dimension"
    )

    padding_image_height = utils.forms.IntegerField(
        u'Padding Image Height',
        default=384,
        validators=[
            validate_required_if_set('padding_image_width'),
            validators.NumberRange(min=1),
        ],
        tooltip="If specified, images will be padded to that dimension"
    )

    channel_conversion = utils.forms.SelectField(
        u'Channel conversion',
        choices=[
            ('RGB', 'RGB'),
            ('L', 'Grayscale'),
            ('none', 'None'),
        ],
        default='RGB',
        tooltip="Perform selected channel conversion."
    )

    val_min_box_size = utils.forms.IntegerField(
        u'Minimum box size (in pixels) for validation set',
        default='25',
        validators=[
            validators.InputRequired(),
            validators.NumberRange(min=0),
        ],
        tooltip="Retain only the boxes that are larger than the specified "
                "value in both dimensions. This only affects objects in "
                "the validation set. Enter 0 to disable this threshold."
    )

    custom_classes = utils.forms.StringField(
        u'Custom classes',
        validators=[
            validators.Optional(),
        ],
        tooltip="Enter a comma-separated list of class names. "
                "Class IDs are assigned sequentially, starting from 0. "
                "Unmapped class names automatically map to 0. "
                "Leave this field blank to use default class mappings. "
                "See object detection extension documentation for more "
                "information."
    )



    #
    # Method - S3
    #

    s3_endpoint_url = utils.forms.StringField(
        u'Training Images',
        tooltip=('S3 end point URL'),
    )

    s3_bucket = utils.forms.StringField(
        u'Bucket Name',
        tooltip=('bucket name'),
    )

    s3_path = utils.forms.StringField(
        u'Training Images Path',
        tooltip=('Indicate a path which holds subfolders full of images. '
                 'Each subfolder should be named according to the desired label for the images that it holds. '),
    )

    s3_accesskey = utils.forms.StringField(
        u'Access Key',
        tooltip=('Access Key to access this S3 End Point'),
    )

    s3_secretkey = utils.forms.StringField(
        u'Secret Key',
        tooltip=('Secret Key to access this S3 End Point'),
    )

    s3_keepcopiesondisk = utils.forms.BooleanField(
        u'Keep Copies of Files on Disk',
        tooltip=('Checking this box will keep raw files retrieved from S3 stored on disk after the job is completed'),
    )

    s3_pct_val = utils.forms.IntegerField(
        u'% for validation',
        default=25,
        validators=[
            validate_required_iff(method='s3'),
            validators.NumberRange(min=0, max=100)
        ],
        tooltip=('You can choose to set apart a certain percentage of images '
                 'from the training images for the validation set.'),
    )

    s3_pct_test = utils.forms.IntegerField(
        u'% for testing',
        default=0,
        validators=[
            validators.NumberRange(min=0, max=100)
        ],
        tooltip=('You can choose to set apart a certain percentage of images '
                 'from the training images for the test set.'),
    )

    s3_train_min_per_class = utils.forms.IntegerField(
        u'Minimum samples per class',
        default=2,
        validators=[
            validators.Optional(),
            validators.NumberRange(min=1),
        ],
        tooltip=('You can choose to specify a minimum number of samples per class. '
                 'If a class has fewer samples than the specified amount it will be ignored. '
                 'Leave blank to ignore this feature.'),
    )

    s3_train_max_per_class = utils.forms.IntegerField(
        u'Maximum samples per class',
        validators=[
            validators.Optional(),
            validators.NumberRange(min=1),
            validate_greater_than('s3_train_min_per_class'),
        ],
        tooltip=('You can choose to specify a maximum number of samples per class. '
                 'If a class has more samples than the specified amount extra samples will be ignored. '
                 'Leave blank to ignore this feature.'),
    )
