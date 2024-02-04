from django.urls import path


from . import views

app_name = 'core'

urlpatterns  = [
    path('',views.index,name='index'),
    path('login/',views.login_user,name='login'),
    path('signup/',views.signup_user,name='signup'),
    path('console/',views.console,name='console'),
    path('logout_user',views.logout_user,name='logout')
]