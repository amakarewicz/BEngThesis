from django.db import models

# Create your models here.


class Variable(models.Model):
    name = models.CharField('Name', max_length=30)
    description = models.CharField('Description', max_length=800)

    def __str__(self):
        return self.name


class Algorithm(models.Model):
    name = models.CharField('Name', max_length=30)
    description = models.CharField('Description', max_length=800)

    def __str__(self):
        return self.name