from django.db import models

from process.models import Process
from django.utils.translation import gettext as _


class Backoffice(models.Model):
    class Type(models.IntegerChoices):
        AUTH = 0, _('AUTH')
        OCR = 1, _('OCR')

    class Status(models.IntegerChoices):
        INIT = 0, _('INIT')
        PENDING = 1, _('Pending')
        COMPLETED = 2, _('Complete')
        FAILED = 3, _('Failed')
        ERROR = 4, _('Error')

    id = models.AutoField(primary_key=True)
    process = models.ForeignKey(Process, on_delete=models.CASCADE, related_name='process', null=True)
    process_res = models.JSONField(null=True)
    process_type = models.IntegerField(choices=Type.choices, default=None)
    process_errors = models.JSONField(null=True, default=None)
    process_status = models.IntegerField(choices=Status.choices, default=None)
    created = models.DateTimeField(editable=False, auto_now_add=True)
    updated = models.DateTimeField(auto_created=True, auto_now=True)

    #