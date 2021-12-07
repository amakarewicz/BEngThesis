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


class Data(models.Model):
    countrycode = models.CharField('countrycode', max_length=3)
    country = models.CharField('country', max_length=30)
    year = models.IntegerField('year')
    pop = models.FloatField('pop')
    rgdpna = models.FloatField('rgdpna')
    delta = models.FloatField('delta')
    xr = models.FloatField('xr')
    csh_c = models.FloatField('csh_c')
    csh_i = models.FloatField('csh_i')
    csh_g = models.FloatField('csh_g')
    csh_x = models.FloatField('csh_x')
    csh_m = models.FloatField('csh_m')
    csh_r = models.FloatField('csh_r')
    rgdpna_per_cap = models.FloatField('rgdpna_per_cap')
    emp_percent = models.FloatField('emp_percent')
    co2_emission = models.FloatField('co2_emission')
    employment_agro = models.FloatField('employment_agro')
    employment_industry = models.FloatField('employment_industry')
    employment_services = models.FloatField('employment_services')
    export_percent = models.FloatField('export_percent')
    import_percent = models.FloatField('import_percent')
    inflation = models.FloatField('inflation')
    net_migration = models.FloatField('net_migration')
    population_15_64 = models.FloatField('population_15_64')
    population_above_65 = models.FloatField('population_above_65')
    population_under_14 = models.FloatField('population_under_14')
    unemployment = models.FloatField('unemployment')
    urban_population = models.FloatField('urban_population')
    hdi = models.FloatField('hdi')

    def __str__(self):
        return self.countrycode


class MetricsValues(models.Model):
    algorithm = models.CharField('algorithm', max_length=50)
    n_clusters = models.IntegerField('n_clusters')
    silhouette = models.FloatField('silhouette', blank=True, null=True)
    chscore = models.FloatField('chscore', blank=True, null=True)
    dunnindex = models.FloatField('dunnindex', blank=True, null=True)

    def __str__(self):
        return self.algorithm