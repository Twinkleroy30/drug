"""
URL configuration for drug project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from mains import views as m
from admins import views as v
from users import views as u


urlpatterns = [
    path('admin/', admin.site.urls),

    path('', m.index, name='index'),
    path('about/', m.about, name='about'),
    path('buy/', m.buy, name='buy'),
    path('contact/', m.contact, name='contact'),
  
    path('medicine/', m.medicine, name='medicine'),

   path('adminlogin/', v.adminlogin, name='adminlogin'),
   path('adashboard/', v.adashboard,name='adashboard'),
   path('cnn/', v.cnn,name='cnn'),
   path('upload/', v.upload, name='upload'),

    path('register/',u.register, name='register'),
    path('userlogin/',u.userlogin, name='userlogin'),
    path('udashboard/', u.udashboard, name='udashboard'),
    path('prediction/', u.prediction, name='prediction'),
    path('medication/', u.medication, name='medication'),
    path('predict/', u.predict, name='predict'),
]
