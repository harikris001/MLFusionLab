from django.shortcuts import render,HttpResponseRedirect
from django.contrib.auth.models import User

def index(request):
    return render(request,'core/index.html')

def login(request):
    return render(request,'core/login.html')

def signup(request):
    context={}
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST['password']
        repassword  = request.POST['repassword']
        email = request.POST['email']
        # print(password,repassword)

        if password == repassword:
            try:
                User.objects.create_user(username = username,password=password,email=email)
                return HttpResponseRedirect('/login/?sucess=true')
            except Exception as e:
                message = "Username already exist"
                context={'error':message,'error_body':"Please enter a different username this already exist"}
                return render(request,'core/signup.html',context=context)
        else:
            message = "Password doesn't match"
            context={'error':message,'error_body':"Passwords that you have entered doesnt match. Try again"}
            return render(request,'core/signup.html',context=context)
    return render(request,'core/signup.html',context=context)

def console(request):
    return render(request,'core/console.html',)