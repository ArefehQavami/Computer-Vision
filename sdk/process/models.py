from django.db import models
from django.utils.translation import gettext as _


class Process(models.Model):
    class Type(models.IntegerChoices):
        AUTH = 0, _('AUTH')
        OCR = 1, _('OCR')

    class Status(models.IntegerChoices):
        INIT = 0, _('INIT')
        PENDING = 1, _('Pending')
        COMPLETED = 2, _('Complete')
        FAILED = 3, _('Failed')
        ERROR = 4, _('Error')

    process_pk = models.AutoField(primary_key=True)
    process_id = models.CharField(max_length=255)
    process_try = models.IntegerField(default=0)
    process_url = models.JSONField(null=True)
    process_type = models.IntegerField(choices=Type.choices)
    process_errors = models.JSONField(null=True)
    process_status = models.IntegerField(choices=Status.choices)
    process_created = models.DateTimeField(editable=False, auto_now_add=True)
    process_updated = models.DateTimeField(auto_created=True, auto_now=True)
    process_json = models.JSONField()

    #
    class Meta:
        ordering = ['-process_created']
