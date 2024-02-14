from django.urls import path
from . import views

urlpatterns = [
    path('dummy/',views.homePage, name='blog'),
    path('', views.indexBlog, name='index'),
    path('products/', views.productsBlog, name='products'),
    path('knife/',views.knifeItem, name='knife'),
    path('table/',views.tableItem, name='table'),
    path('chair/',views.chairItem, name='chair'),
    # path ('machine/', views.machine_learning),
    # path('dt/', views.decision),
    # path('knn/', views.knn)
]