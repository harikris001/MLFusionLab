from django.shortcuts import render
from django_tables2.tables import Table
import pandas as pd
from . models import TableDataFile

def tag(request):
    if request.method == 'POST':
        csvfile = request.FILES.get('csv_file')
        data = pd.read_csv(csvfile)
        # instance = TableDataFile(csv_file = request.FILES.get('csv_file'))
        # instance.save()
        fields = list(data.columns)
        data = data.to_html(max_rows=6,classes=['table px-4 border border-info-subtle rounded'],justify='center')
        context = {'df_table': data,'columns': fields}
        return render(request, 'tabular/tag.html', context)
    return render(request, 'tabular/tag.html')

def tabular(request):
    return render(request,'tabular/tabular.html')