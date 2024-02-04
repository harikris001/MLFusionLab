from django.shortcuts import render, HttpResponseRedirect, HttpResponse
from . models import Project
from . training_backend import train_models, best_model, train_best_model

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.preprocessing import LabelEncoder
import shutil
import os


def regression(request):
    if request.method == 'GET':
        pid = request.GET.get('pid')
        file_path = Project.objects.filter(pk=pid).first().csv_file
        df = pd.read_csv(file_path)
        fields = list(df.columns)
        df = df.to_html(max_rows=6,classes=['table px-4 border border-info-subtle rounded'],justify='center')
        context = {'df_table': df,'columns': fields,'pid':pid}
        return render(request, 'tabular/regression.html', context)
    return render(request, 'tabular/regression.html')

def tabular(request):
    if request.method == 'POST':
        pjt_name = request.POST.get('project-name')
        desc = request.POST.get('project-description')
        csv_file = request.FILES['csv_file']
        pjt = Project(name = pjt_name, description = desc, csv_file = csv_file)
        pjt.save()
        return HttpResponseRedirect(f'regression/?pid={pjt.pk}')
    return render(request,'tabular/tabular.html')
        

def training(request):
    if request.method == 'POST':
        project_id = request.GET.get('pid')
        filepath = Project.objects.filter(pk=project_id).first().csv_file
        print(filepath)
        data = pd.read_csv(filepath)
        data.drop_duplicates(inplace= True)
        data.dropna(inplace=True)
        cols = list(data.columns)
        lab = LabelEncoder()
        obdata,numdata = [],[]
        for col in cols:
            val = request.POST.get(col)
            if val == 'string':
                obdata.append(col)
            elif val == 'integer' or val == 'float':
                numdata.append(col)
            else:
                data = data.drop([col],axis=1)
        data[obdata] = data[obdata].apply(lab.fit_transform)
        target = request.POST.get('target')
        x= data.drop(target,axis=1)
        y= data[target]
        algorithm = ['LinearRegression','DecisionTreeClassifier','RandomForestClassifier','GradientBoostingRegressor','SVR']
        model1 = LinearRegression()
        model2 = DecisionTreeRegressor()
        model3 = RandomForestRegressor()
        model4 = GradientBoostingRegressor()
        model5 = SVR()
        models_list = [model1, model2, model3, model4, model5]
        R2=[]
        RMSE = []
        for model in models_list:
            r2, rmse = train_models(model,x,y)
            R2.append(r2)
            RMSE.append(rmse)
        output_df = pd.DataFrame({'Algorithm':algorithm, 'R2_Score': R2, 'RMSE':RMSE})
        output_df = output_df.to_html(max_rows=6,classes=['table px-4 border border-info-subtle rounded'],justify='center')
        best, max_score, avg_error = best_model(R2,RMSE)
        train_best_model(model_list=models_list,index=best,x=x,y=y)
        context = {'output_df': output_df,'best_model':algorithm[best],'score':max_score,'rmse':avg_error}
        return render(request,'tabular/results.html', context)

def download_models(request):
    origin = 'RFmodel.pkl'
    target = 'models/'
    
    shutil.move(origin, target+origin)
    
    best_model = 'models/RFmodel.pkl'
    response = HttpResponse(best_model, content_type='application/force-download')
    response['Content-Disposition'] = f'attachment; filename="RFmodel.pkl"'
    return response