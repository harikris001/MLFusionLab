from django.urls import path
from . import views

app_name = 'text_model'

urlpatterns = [
    path('',views.tabular,name='tabular'),
    path('regression/',views.regression,name='regression'),
    path('results/',views.training,name='training'),
]