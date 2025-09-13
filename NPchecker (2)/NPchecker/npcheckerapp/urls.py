from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('check-results/', views.get_results, name='get_results'),
    path('history/', views.history, name='history'),
   
     path('about/', views.about, name='about'),
    path('results/', views.results, name='results'),
    path('coat_of_arm/', views.coat_of_arm, name='coat_of_arm'),
      path('get-results/', views.get_results, name='get_results'), 
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)