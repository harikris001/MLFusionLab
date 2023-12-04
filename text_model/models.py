from django.db import models

class Project(models.Model):
    name = models.CharField(max_length=256)
    description = models.TextField(blank=True)
    csv_file = models.FileField(upload_to='uploads/')

