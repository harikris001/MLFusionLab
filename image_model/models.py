from django.db import models

class Project(models.Model):
    name = models.CharField(max_length=256)
    description = models.TextField(blank=True)
    dataset = models.FileField(upload_to="uploads/Image_datasets")
    pjt_type = models.CharField(max_length=20)