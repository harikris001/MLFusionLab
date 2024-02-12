from django.urls import path


from . import views

app_name = 'image_model'

urlpatterns  = [
    path('',views.image,name='image'),
    path('detection/',views.detection,name='detection'),
    path('classification/',views.classify, name='classification'),
    path('segmentation/',views.segmentation, name='segmentation'),
    path('results/',views.training,name='results')
]