from django.shortcuts import render,HttpResponseRedirect, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required

def index(request):
    return render(request,'core/index.html')

def login_user(request):
    if request.method == 'GET':
        success = request.GET.get('success')
        if success == 'true':
            context = {'message':'Signup successfull'}
            return render(request,'core/login.html',context)
    context = {}
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('/console')
        else:
            context = {'message':'username or password wrong'}
            return render(request,'core/login.html',context)
    return render(request,'core/login.html')
def signup_user(request):
    context={}
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST['password']
        repassword  = request.POST['repassword']
        email = request.POST['email']

        if password == repassword:
            try:
                User.objects.create_user(username = username,password=password,email=email)
                return HttpResponseRedirect('/login/?success=true')
            except Exception as e:
                message = "Username already exist"
                context={'error':message,'error_body':"Please enter a different username this already exist"}
                return render(request,'core/signup.html',context=context)
        else:
            message = "Password doesn't match"
            context={'error':message,'error_body':"Passwords that you have entered doesnt match. Try again"}
            return render(request,'core/signup.html',context=context)
    return render(request,'core/signup.html',context=context)

def logout_user(request):
    logout(request)
    return redirect('/')

@login_required
def console(request):
    return render(request,'core/console.html',)