from django.urls import path
from . import views

app_name = 'text_model'

urlpatterns = [
    path('',views.hello,name='text_model')
]