from django.db import models

class TableDataFile(models.Model):
    csv_file = models.FileField()