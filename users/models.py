from django.db import models
from django.db import models
from django.db import models

class User(models.Model):
    name = models.CharField(max_length=100)
    mobile = models.CharField(max_length=15)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=100)
    age = models.IntegerField()
    address = models.TextField()
    profile_picture = models.ImageField(upload_to='profiles/', blank=True, null=True)  # Image upload field

    def __str__(self):
        return self.name
# Create your models here.
# forms.py
