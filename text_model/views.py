from django.shortcuts import render

def hello(request):
    return render(request,'text_model/tag.html')
